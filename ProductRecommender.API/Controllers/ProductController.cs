using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using Microsoft.ML;
using ProductRecommender.Models;
using ProductRecommender.Models.Dtos;

namespace ProductRecommender.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ProductController : ControllerBase
    {
        private readonly PredictionEnginePool<ProductRating, ProductRatingPrediction> _predictionEnginePool;
        public ProductController(PredictionEnginePool<ProductRating, ProductRatingPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        // GET: api/<ProductController>
        [HttpGet]
        public IEnumerable<string> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET api/<ProductController>/5
        [HttpGet("{id}")]
        public string Get(int id)
        {
            return "value";
        }

        [HttpPost("predict")]
        public IActionResult Predict([FromBody] ProductRatingInput inputData)
        {
            var rating = new ProductRating
            {
                user_id = inputData.UserId,
                product_id = inputData.ProductId,
                Label = 0f
            };

            PredictionEngine<ProductRating, ProductRatingPrediction> engineByName =
                _predictionEnginePool.GetPredictionEngine("ProductRecommenderModel");

            var prediction = engineByName.Predict(rating);

            return Ok(new
            {
                prediction.Score
            });
        }
    }
}
