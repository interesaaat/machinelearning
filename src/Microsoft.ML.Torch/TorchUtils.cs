// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Data.DataView;
using TorchSharp.Tensor;

namespace Microsoft.ML.Torch
{
    public static class TorchUtils
    {
        /// <summary>
        /// Key to access operator's type (a string) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value describes the Torch operator that produces this <see cref="DataViewSchema.Column"/>.
        /// </summary>
        internal const string TorchOperatorTypeKind = "TorchOperatorType";
        /// <summary>
        /// Key to access upstream operators' names (a string array) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value states operators that the associated <see cref="DataViewSchema.Column"/>'s generator depends on.
        /// </summary>TorchSharp
        internal const string TorchUpstreamOperatorsKind = "TorchUpstreamOperators";

        /// <summary>
        /// Load Torch model into memory.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">The model to load.</param>
        /// <returns></returns>
        internal static TorchModel LoadTorchModel(IHostEnvironment env, string modelPath)
        {
            var module = GetModule(env, modelPath);
            return new TorchModel(env, module, modelPath);
        }

        internal static TorchSharp.JIT.Module GetModule(IHostEnvironment env, string modelPath)
        {
            Contracts.Assert(CheckModel(env, modelPath));
            return TorchSharp.JIT.Module.Load(modelPath);
        }

        internal static bool CheckModel(IHostEnvironment env, string modelPath)
        {
            Contracts.CheckValue(env, nameof(env));
            if (IsTorchScriptModel(env, modelPath))
            {
                return true;
            }

            return false;
        }

        // A PyTorch TorchScript model is a single file. Given a modelPath, this utility method
        // determines if we should treat it as a TorchScript model or not.
        internal static bool IsTorchScriptModel(IHostEnvironment env, string modelPath)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(modelPath, nameof(modelPath));
            env.CheckUserArg(File.Exists(modelPath), nameof(modelPath));
            FileAttributes attr = File.GetAttributes(modelPath);
            return attr.HasFlag(FileAttributes.Archive);
        }

        internal static DataViewSchema GetModelSchema(IExceptionContext ectx, TorchSharp.JIT.Module module, string opType = null)
        {
            throw new NotImplementedException();
        }

        internal static PrimitiveDataViewType Tf2MlNetType(ATenScalarMapping type)
        {
            var mlNetType = Torch2MlNetTypeOrNull(type);
            if (mlNetType == null)
                throw new NotSupportedException("Torch type not supported.");
            return mlNetType;
        }

        private static PrimitiveDataViewType Torch2MlNetTypeOrNull(ATenScalarMapping type)
        {
            switch (type)
            {
                case ATenScalarMapping.Float:
                    return NumberDataViewType.Single;
                case ATenScalarMapping.Double:
                    return NumberDataViewType.Double;
                case ATenScalarMapping.Byte:
                    return NumberDataViewType.Byte;
                case ATenScalarMapping.Int:
                    return NumberDataViewType.Int32;
                case ATenScalarMapping.Long:
                    return NumberDataViewType.Int64;
                case ATenScalarMapping.Short:
                    return NumberDataViewType.Int16;
                default:
                    return null;
            }
        }
    }
}
