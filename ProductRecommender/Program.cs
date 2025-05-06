using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using ProductRecommender.Models;

namespace ProductRecommender;

class Program
{
    static void Main(string[] args)
    {
        // Create MLContext to be shared across the model creation workflow objects
        MLContext mlContext = new MLContext();

        // Load data
        (IDataView training, IDataView test) = LoadData(mlContext);

        var trainSamples = mlContext.Data.CreateEnumerable<ProductRating>(training, reuseRowObject: false).Take(5);
        foreach (var row in trainSamples)
        {
            Console.WriteLine($"User: {row.user_id}, Movie: {row.product_id}, Rating: {row.Label}");
        }

        // Build & train model
        ITransformer model = BuildAndTrainModel(mlContext, training);

        // Evaluate quality of model
        EvaluateModel(mlContext, test, model);

        // Use model to try a single prediction (one row of data)
        UseModelForSinglePrediction(mlContext, model);

        // Save model
        SaveModel(mlContext, training.Schema, model);
    }

    public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
    {
        var rawPath = Path.Combine(Environment.CurrentDirectory, "Data", "amazon.csv");
        var cleanedData = new List<ProductRating>();

        using (var reader = new StreamReader(rawPath))
        using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
               {
                   HasHeaderRecord = true,
                   BadDataFound = null
               }))
        {
            while (csv.Read())
            {
                try
                {
                    var productId = csv.GetField(0)?.Replace(",", " ");
                    var rating = csv.GetField(6);
                    var userId = csv.GetField(9)?.Replace(",", " ");

                    if (!string.IsNullOrWhiteSpace(productId) &&
                        !string.IsNullOrWhiteSpace(userId) &&
                        float.TryParse(rating, out float parsedRating))
                    {
                        cleanedData.Add(new ProductRating
                        {
                            product_id = productId,
                            user_id = userId,
                            Label = parsedRating
                        });
                    }
                }
                catch
                {
                    continue; // Skip bad rows
                }
            }
        }

        // Load in-memory data to IDataView
        IDataView data = mlContext.Data.LoadFromEnumerable(cleanedData);

        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        return (split.TrainSet, split.TestSet);
    }

    public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
    {
        IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion
            .MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "user_id")
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "productIdEncoded",
                inputColumnName: "product_id"));
        var options = new MatrixFactorizationTrainer.Options
        {
            MatrixColumnIndexColumnName = "userIdEncoded",
            MatrixRowIndexColumnName = "productIdEncoded",
            LabelColumnName = "Label",
            NumberOfIterations = 20,
            ApproximationRank = 100
        };

        var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

        ITransformer model = trainerEstimator.Fit(trainingDataView);

        return model;
    }

    public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
    {
        // Evaluate model on test data & print evaluation metrics
        Console.WriteLine("=============== Evaluating the model ===============");
        var prediction = model.Transform(testDataView);

        var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

        Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
        Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
    }

    // Use model for single prediction
    public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
    {
        Console.WriteLine("=============== Making a prediction ===============");
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductRating, ProductRatingPrediction>(model);

        // Create test input & make single prediction
        var testInput = new ProductRating
        {
            user_id = "AGYLPKPZHVYKKZHOTHCTYVEDAJ4A AGTTU64JMX722LYCN3SOWLFPKPAQ AFWD4ZTM7473CDWARHCDQKK73MTA AEXCQM3FDLX3YL3UJWWUIAIUJT4A AHUKYUWRUVRTB3IQGISXWTSPAWLQ AFWW4UEXAJH7EAB5LTMKMSGLUN2Q AFM5JL37WY7G6MLQUI4WAXUJME7Q AFECO24WYFOU2KL7C3DMHTEHRU7Q",
            product_id = "B01486F4G6"
        };

        var movieRatingPrediction = predictionEngine.Predict(testInput);

        if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
        {
            Console.WriteLine("Product " + testInput.product_id + " is recommended for user " + testInput.user_id);
        }
        else
        {
            Console.WriteLine("Product " + testInput.product_id + " is not recommended for user " + testInput.user_id);
        }
    }

    //Save model
    public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
    {
        // Save the trained model to .zip file
        var projectRoot = Directory.GetParent(Environment.CurrentDirectory)?.Parent?.Parent?.FullName;
        var modelPath = Path.Combine(projectRoot ?? string.Empty, "Data", "ProductRecommenderModel.zip");

        Console.WriteLine("=============== Saving the model to a file ===============");
        mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
    }
}