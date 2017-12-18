using RF2;
using RF2.Entities;
using RF2.Evaluation;
using RF2.Readers;
using RF2.Recommenders;
using System;
using MyMediaLite.RatingPrediction;
using MyMediaLite.ItemRecommendation;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {

            // step 1: dataset            
            var dataset = new Dataset<MovieLensItemRating>(new MovieLensReader("/2/ratings.dat"), 0.7);

            // step 2: recommender
            var recommender = new MediaLiteRatingPredictor(new BiasedMatrixFactorization());

            //Teste
            //var recommender = new LibFmTrainTester();


            //var obj = Newtonsoft.Json.JsonConvert.SerializeObject(dataset);
            //Console.WriteLine(obj);

            // step3: evaluation
            var ep = new EvaluationPipeline<ItemRating>(new EvalutationContext<ItemRating>(recommender, dataset));
            ep.Evaluators.Add(new RMSE());
            ep.Evaluators.Add(new MAE());

            ep.Run();

            Console.WriteLine("Finished!.");
            Console.ReadLine();

        }
    }
}
