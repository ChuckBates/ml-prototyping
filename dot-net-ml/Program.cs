using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

// CS0649 compiler warning is disabled because some fields are only
// assigned to dynamically by ML.NET at runtime
#pragma warning disable CS0649

namespace myMLApp
{
    class Program
    {
        public class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        // IrisPrediction is the result returned from prediction operations
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel;
        }

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(path: "iris-data.txt", hasHeader: false, separatorChar: ',');
            
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, pipeline, 10);
            var accuracies = crossValidationResults.Select(r => r.Metrics.MicroAccuracy);
            Console.WriteLine($"Cross validation produced a {accuracies.Average():F} average accuracy");

            var model = pipeline.Fit(trainingDataView);

            var prediction = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model).Predict(
                new IrisData()
                {
                    SepalLength = 5.5f,
                    SepalWidth = 2.4f,
                    PetalLength = 3.8f,
                    PetalWidth = 1.1f,
                });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
            Console.ReadKey();
        }
    }
}