using Microsoft.ML.Data;

namespace ProductRecommender.Models
{
    public class ProductRating
    {
        [LoadColumn(0)]
        public string product_id;
        [LoadColumn(9)]
        public string user_id;
        [LoadColumn(6)]
        public float Label;
    }
}
