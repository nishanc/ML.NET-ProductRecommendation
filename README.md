# 🧠 ML.NET Product Recommendation Demo – Co-Purchase Scenario

This demo accompanies the tech talk **"Building Intelligent Applications with ML.NET"**, where we explore how to use **Matrix Factorization** to build a **product recommendation system** based on co-purchase behavior.

---

## 📌 Scenario

Imagine you're running an e-commerce platform. You have customer purchase data and want to recommend products that are **frequently bought together** — even if the user hasn't seen them before.

This solution uses:
- ML.NET’s `MatrixFactorizationTrainer`
- A small sample dataset of `user_id`, `product_id`, and `Label`

---

## 💡 What You'll Learn

- Basics of Matrix Factorization for collaborative filtering
- How to implement recommendations in ML.NET
- Training a model using implicit purchase data
- Making product predictions for a given customer

---

## 🔧 Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download)
- Visual Studio or VS Code
- ML.NET NuGet Package:

```
dotnet add package Microsoft.ML
```

## 🛠️ How to Run the Demo

1. **Clone or download the repo**.
2. Open the folder in Visual Studio or run via CLI.
3. Make sure the dataset `amazon.csv` is in the `Data/` folder.
4. Run the program:
 ```bash
 dotnet run
 ```

 
## 🔗 Further Reading and References

### 📦 ML.NET Samples

- 🔹 [Product Recommendation - Matrix Factorization Problem Sample](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/MatrixFactorization_ProductRecommendation#product-recommendation---matrix-factorization-problem-sample)
- 🎬 [Movie Recommendation - Matrix Factorization Sample 1 (Program.cs)](https://github.com/dotnet/samples/blob/main/machine-learning/tutorials/MovieRecommendation/Program.cs)
- 🎬 [Movie Recommendation - Matrix Factorization Sample 2](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/MatrixFactorization_MovieRecommendation#movie-recommendation---matrix-factorization-problem-sample)

### 📘 Documentation and Tutorials

- 📄 [What is ML.NET and How Does It Work?](https://learn.microsoft.com/en-us/dotnet/machine-learning/mldotnet-api)
- ⚙️ [What is Automated Machine Learning (AutoML)?](https://learn.microsoft.com/en-us/dotnet/machine-learning/automated-machine-learning-mlnet)
- 🛠️ [ML.NET AutoML Model Builder (Step-by-Step Walkthrough)](https://blog.nishanc.com/2023/08/mlnet-automl-model-builder-step-by-step.html)
- 🚀 [Deploy a Model in an ASP.NET Core Web API](https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/serve-model-web-api-ml-net)

### 📊 Data & Research

- 🛒 [Amazon Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset?resource=download)
- 📑 [Machine Learning at Microsoft with ML.NET (Research Paper)](https://arxiv.org/pdf/1905.05715)

### 🤖 Ecosystem & Showcase

- 🌐 [Open Neural Network Exchange (ONNX)](https://onnx.ai/)
- 🏢 [Artificial Intelligence & ML Customer Showcase (Microsoft)](https://dotnet.microsoft.com/en-us/platform/customers/machinelearning-ai)
