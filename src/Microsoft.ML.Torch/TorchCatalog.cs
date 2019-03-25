// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Torch;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Transforms.TorchTransformer;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="TorchTransformer"]/*' />
    public static class TorchCatalog
    {
        /// <summary>
        /// Load TensorFlow model into memory. This is the convenience method that allows the model to be loaded once
        /// and subsequently use it for querying schema and creation of
        /// <see cref="TorchEstimator"/> using <see cref="TorchModel.ScoreTorchModel(long[])"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        public static TorchModel LoadTorchModel(this ModelOperationsCatalog catalog, string modelLocation)
            => TorchUtils.LoadTorchModel(CatalogUtils.GetEnvironment(catalog), modelLocation);
    }
}
