// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using static Microsoft.ML.Transforms.TorchTransformer;

namespace Microsoft.ML.Torch
{
    /// <summary>
    /// This class holds the information related to Torch model.
    /// It provides some convenient methods to query the model schema as well as
    /// creation of <see cref="TorchEstimator"/> object.
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

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for complete model.
        /// Every node in the Torch model will be included in the <see cref="DataViewSchema"/> object.
        /// </summary>
        public DataViewSchema GetModelSchema()
        {
            //return TorchUtils.GetModelSchema(_env);
            return null;
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
        public TorchEstimator ScoreTorchModel(long[] shape)
            => new TorchEstimator(_env, shape, this);
    }
}