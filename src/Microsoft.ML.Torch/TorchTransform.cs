using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Data.DataView;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Torch;
using TorchSharp.Tensor;

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="TorchTransformer"]/*' />
    public sealed class TorchTransformer : RowToRowTransformerBase
    {
        internal readonly string Input;
        internal readonly string Output;
        internal readonly long[] InputShape;
        internal readonly TorchSharp.JIT.Module Module;

        internal static int BatchSize = 1;
        internal const string Summary = "Transforms the data using the Torch model.";
        internal const string UserName = "TorchTransform";
        internal const string ShortName = "TOTransform";
        internal const string LoaderSignature = "TorchTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TORCH",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TorchTransformer).Assembly.FullName);
        }

        internal TorchTransformer(
            IHostEnvironment env,
            TorchSharp.JIT.Module module,
            string outputColumnName,
            string inputColumnName,
            long[] inputShape) :
                base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchTransformer)))

        {
            Host.CheckValue(module, nameof(module));
            Host.CheckNonWhiteSpace(inputColumnName, nameof(inputColumnName));
            Host.CheckNonWhiteSpace(outputColumnName, nameof(outputColumnName));

            Module = module;
            Input = inputColumnName;
            InputShape = inputShape;
            Output = outputColumnName;
        }

        ~TorchTransformer()
        {
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                Module.Dispose();
            }
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            throw new System.NotImplementedException();
        }

        private sealed class Mapper : MapperBase
        {
            private readonly TorchTransformer _parent;
            private readonly int _inputColIndex;
            private readonly bool _isInputVector;
            private readonly bool _needReshape;
            private readonly long[] _inputShape;

            public Mapper(TorchTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;

                if (!inputSchema.TryGetColumnIndex(_parent.Input, out _inputColIndex))
                {
                    throw Host.ExceptSchemaMismatch(nameof(InputSchema), "source", _parent.Input);
                }

                var type = inputSchema[_inputColIndex].Type;
                if (type is VectorType vecType && vecType.Size == 0)
                {
                    throw Host.Except("Variable length input columns not supported");
                }
                //if (type.GetItemType() != DataViewType.)
                //{
                //    throw Host.Except($"Input type does not match with the expected type");
                //}

                _isInputVector = type is VectorType;
                vecType = (VectorType)type;
                _inputShape = _parent.InputShape;

                var colTypeDims = vecType.Dimensions.Select(dim => (long)dim).ToArray();
                var colShapeLength = colTypeDims.Length;
                if (colShapeLength > 1)
                {
                    if (colShapeLength != _inputShape.Length)
                    {
                        throw Host.Except($"Input shape length {colShapeLength} and expected shape length {_inputShape.Length} do not match.");
                    }

                    for (int i = 0; i < colShapeLength; i++)
                    {
                        if (colTypeDims[i] != _inputShape[i])
                        {
                            throw Host.Except($"Input dimension {i} of size {colTypeDims[i]} do not match expected size {_inputShape[i]}.");
                        }
                    }
                }
                else
                {
                    // If the column is one dimension we make sure that the total size of the TF shape matches.
                    long valCount = _inputShape.Aggregate(1, (long x, long y) => x * y);

                    if (vecType.Size != valCount)
                    {
                        throw Host.Except($"Input size {vecType.Size} does not match with expect size {valCount}.");
                    }

                    _needReshape = true;
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return new DataViewSchema.DetachedColumn[]
                {
                    new DataViewSchema.DetachedColumn(_parent.Output, new VectorType(NumberDataViewType.Single, _inputShape.Select(s => (int)s).ToArray()), null)
                };
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);

                ITorchTensor<float> tensor = null;

                ValueGetter<VBuffer<float>> valuegetter = (ref VBuffer<float> dst) =>
                {
                    tensor = CreateTensorValueGetter<float>(input, _isInputVector, _inputColIndex, _inputShape);

                    ITorchTensor<float> result = _parent.Module.Forward(tensor);
                    var resultSize = result.NumberOfElements;
                    var editor = VBufferEditor.Create(ref dst, (int)resultSize);

                    result.Data.CopyTo(editor.Values);
                    dst = editor.Commit();
                    result.Dispose();
                };

                disposer = () =>
                {
                    tensor?.Dispose();
                };

                return valuegetter;
            }

            private ITorchTensor<T> CreateTensorValueGetter<T>(DataViewRow input, bool isVector, int colIndex, long[] shape)
            {
                if (isVector)
                {
                    var srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                    VBuffer<T> vBuffer = default;
                    T[] denseData = default;
                    GCHandle handle;
                    ITorchTensor<T> tensor = default;

                    srcgetter(ref vBuffer);

                    // _denseData.Length can be greater than _vBuffer.Length sometime after
                    // Utils.EnsureSize is exectued. Use _vBuffer.Length to access the elements in _denseData.
                    // This is done to reduce memory allocation every time tensor is created.
                    Utils.EnsureSize(ref denseData, vBuffer.Length, keepOld: false);
                    vBuffer.CopyTo(denseData);

                    tensor = denseData.ToTorchTensor(shape);

                    if (handle.IsAllocated)
                    {
                        handle.Free();
                    }

                    if (_needReshape)
                    {
                        var result = tensor.View(_inputShape);
                        tensor.Dispose(); // Need to dipose the partial tensor here otherwise we will have a memory leack
                        return result;
                    }

                    return tensor;
                }
                else
                {
                     var srcgetter = input.GetGetter<T>(colIndex);
                     var scalar = default(T);

                     srcgetter(ref scalar);
                     return scalar.ToTorchTensor();
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, 1).Any(i => activeOutput(i)) && _inputColIndex == col;
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Estimator for the <see cref="TorchTransformer"/>.
        /// </summary>
        public sealed class TorchEstimator : IEstimator<TorchTransformer>
        {
            /// <summary>
            /// The options for the <see cref="TorchTransformer"/>.
            /// </summary>
            internal sealed class Options : TransformInputBase
            {
                /// <summary>
                /// The shape of the model input.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The shape of the input tensor", ShortName = "shape", SortOrder = 0)]
                public long[] InputShape;
            }

            private readonly IHost _host;
            private readonly TorchModel _torchModel;
            private readonly string _inputColumn;
            private readonly string _outputColumn;
            private readonly long[] _inputShape;
            private TorchTransformer _transformer;

            internal TorchEstimator(IHostEnvironment env, long[] shape, TorchModel torchModel)
                : this(env, CreateArguments(shape), torchModel)
            {
            }

            internal TorchEstimator(IHostEnvironment env, Options options, TorchModel torchModel)
            {
                _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchEstimator));
                _torchModel = torchModel;

                _inputColumn = "Features";
                _outputColumn = "Features";

                if (options.InputShape.Any(x => x <= 0))
                {
                    throw _host.Except("Unknown shape dimensions not supported.");
                }
                _inputShape = options.InputShape;
            }

            private static Options CreateArguments(long[] shape)
            {
                var options = new Options();
                options.InputShape = shape;
                return options;
            }

            public TorchTransformer Fit(IDataView input)
            {
                _host.CheckValue(input, nameof(input));
                if (_transformer == null)
                {
                    _transformer =  new TorchTransformer(_host, _torchModel.Module, _outputColumn, _inputColumn, _inputShape);
                }
                // Validate input schema.
                _transformer.GetOutputSchema(input.Schema);
                return _transformer;
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                _host.CheckValue(inputSchema, nameof(inputSchema));
                var resultDic = inputSchema.ToDictionary(x => x.Name);

                if (!inputSchema.TryFindColumn(_inputColumn, out var col))
                {
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _inputColumn);
                }
                if (col.Kind != SchemaShape.Column.VectorKind.Vector)
                {
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _inputColumn, "vector", col.GetTypeString());
                }

                resultDic[_outputColumn] = new SchemaShape.Column(
                    _outputColumn,
                    SchemaShape.Column.VectorKind.VariableVector,
                    NumberDataViewType.Single,
                    false);

                return new SchemaShape(resultDic.Values);
            }
        }
    }
}
