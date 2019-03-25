using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework.Attributes;
using TorchSharp.Tensor;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios
{
    public partial class ScenariosTests
    {
        private class MINSTInputData
        {
            [VectorType(1, 3, 224, 224)]
            public float[] Features { get; set; }
        }

        private class MINSTOutputData
        {
            [VectorType(1000)]
            public float[] Features { get; set; }
        }

        [TorchFact]
        public void TorchMNISTScoringTest()
        {
            var mlContext = new MLContext();
            var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });
            var data = new MINSTInputData
            {
                Features = ones.Data.ToArray()
            };
            var dataPoint = new List<MINSTInputData>() { data };

            var dataView = mlContext.Data.LoadFromEnumerable(dataPoint);

            var output = mlContext.Model
                .LoadTorchModel(@"E:\Source\Repos\libtorch\model.pt")
                .ScoreTorchModel(new long[] { 1, 3, 224, 224 })
                .Fit(dataView)
                .Transform(dataView);

             var count = mlContext.Data.CreateEnumerable<MINSTOutputData>(output, false).Count();
            Assert.True(count == 1000);
        }
    }
}
