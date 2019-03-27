// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Torch
{
    /// <summary>
    /// This class holds the information related to Torch model.
    /// It provides some convenient methods to query the model schema as well as
    /// creation of <see cref="TorchScoringEstimator"/> object.
    /// </summary>
    public sealed class TorchModel
    {
        internal string ModelPath { get; }
        internal TorchSharp.JIT.Module Module { get; }

        private readonly IHostEnvironment _env;

        /// <summary>
        /// Instantiates <see cref="TorchModel"/>.
        /// </summary>
        /// <param name="env">An <see cref="IHostEnvironment"/> object.</param>
        /// <param name="module">The TorchSharp module object containing the model</param>
        /// <param name="modelLocation">Location of the model.</param>
        internal TorchModel(IHostEnvironment env, TorchSharp.JIT.Module module, string modelLocation)
        {
            _env = env;
            Module = module;
            ModelPath = modelLocation;
        }

        // REVIEW: we need to figure out what this was doing in tensorflow models and whether it makes sense to have in torch models.
        // I guess it's a utility method to inspect the schema of the model when you load the model. Is it the input schema or the output schema?
        public DataViewSchema GetModelSchema()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Scores a dataset using a pre-traiend <a href="https://www.pytorch.org/">Torch</a> model.
        /// </summary>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTorchModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TorchTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TorchScoringEstimator ScoreTorchModel(string outputColumnName, long[][] shapes, string[] inputColumnNames = null)
        {
            var options = new TorchScoringEstimator.Options {
                OutputColumnName = outputColumnName,
                InputColumnNames = inputColumnNames ?? new[] { outputColumnName },
                InputShapes = shapes,
                ModelLocation = ModelPath
            };
            return new TorchScoringEstimator(_env, options, this);
        }

        /// <summary>
        /// Scores a dataset using a pre-traiend <a href="https://www.pytorch.org/">Torch</a> model.
        /// </summary>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTorchModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TorchTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TorchScoringEstimator ScoreTorchModel(string outputColumnName, long[] shape, string inputColumnNames = null)
        {
            var options = new TorchScoringEstimator.Options
            {
                OutputColumnName = outputColumnName,
                InputColumnNames = new[] { inputColumnNames } ?? new[] { outputColumnName },
                InputShapes = new[] { shape },
                ModelLocation = ModelPath
            };
            return new TorchScoringEstimator(_env, options, this);
        }
    }
}