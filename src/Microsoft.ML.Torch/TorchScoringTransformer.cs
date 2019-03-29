using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Torch;
using Microsoft.ML.Transforms;
using TorchSharp.Tensor;

[assembly: LoadableClass(TorchScoringTransformer.Summary, typeof(IDataTransform), typeof(TorchScoringTransformer), typeof(TorchScoringEstimator.Options), typeof(SignatureDataTransform),
    TorchScoringTransformer.UserName)]

[assembly: LoadableClass(TorchScoringTransformer.Summary, typeof(IDataTransform), typeof(TorchScoringTransformer), null, typeof(SignatureLoadDataTransform),
    TorchScoringTransformer.UserName, TorchScoringTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(TorchScoringTransformer), null, typeof(SignatureLoadModel),
    TorchScoringTransformer.UserName, TorchScoringTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TorchScoringTransformer), null, typeof(SignatureLoadRowMapper),
    TorchScoringTransformer.UserName, TorchScoringTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public sealed class TorchScoringTransformer : RowToRowTransformerBase
    {
        private readonly string _savedModelPath;

        internal readonly string OutputColumnName;
        internal readonly string[] InputColumnNames;
        internal readonly long[][] InputShapes;
        internal readonly TorchSharp.JIT.Module Module;

        internal const string Summary = "Transforms the data using a Torch model.";
        internal const string UserName = "TorchTransform";
        internal const string LoaderSignature = "TorchTransform";
        private const string _modelFileRepo = "TorchJITModule";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TORCHSCO", // Torch scoring
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TorchScoringTransformer).Assembly.FullName);
        }

        internal TorchScoringTransformer(IHostEnvironment env, TorchSharp.JIT.Module module, string outputColumnName, string[] inputColumnNames,
            long[][] inputShape, string savedModelPath)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchScoringTransformer)))
        {
            Host.CheckValue(module, nameof(module));
            Host.CheckNonWhiteSpace(outputColumnName, nameof(outputColumnName));
            Host.CheckNonWhiteSpace(savedModelPath, nameof(savedModelPath));
            Host.CheckValue(inputColumnNames, nameof(inputColumnNames));
            Host.Check(!inputColumnNames.Any(x => x == null), "Input column names cannot not be null.");
            Host.CheckValue(inputShape, nameof(inputShape));

            OutputColumnName = outputColumnName;
            InputColumnNames = inputColumnNames;
            InputShapes = inputShape;
            Module = module;
            _savedModelPath = savedModelPath;
        }

        // Factory method for SignatureLoadModel.
        internal TorchScoringTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchScoringTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // string output column name
            // int: number of input columns
            // for each input column
            //   string: input column name
            //   int: length of the inputshape array for this input column
            //   for each element in the inputshape
            //      long: element
            // stream: torch JIT module

            OutputColumnName = ctx.LoadNonEmptyString();

            var numInputs = ctx.Reader.ReadInt32();
            Host.CheckDecode(numInputs > 0);
            InputColumnNames = new string[numInputs];
            InputShapes = new long[numInputs][];
            for (int i = 0; i < InputColumnNames.Length; i++)
            {
                InputColumnNames[i] = ctx.LoadNonEmptyString();
                var inputShapeLength = ctx.Reader.ReadInt32();
                Host.CheckDecode(inputShapeLength > 0);
                InputShapes[i] = new long[inputShapeLength];
                for (int j = 0; j < InputShapes[i].Length; j++)
                {
                    InputShapes[i][j] = ctx.Reader.ReadInt64();
                    Host.CheckDecode(InputShapes[i][j] > 0);
                }
            }

            // Creates a temporary directory with a file containing the model. We can the load the model using the Torch::JIT::Module::Load() method that takes a path.
            // REVIEW: ideally we can load a module directly from a stream without the trick of a temp directory. This needs to be added to TorchSharp.
            var tempDirPath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), nameof(TorchScoringTransformer) + "_" + Guid.NewGuid()));
            TorchUtils.CreateFolder(Host, tempDirPath);
            try
            {
                string fullFilePath = null;
                var load = ctx.TryLoadBinaryStream(_modelFileRepo, reader =>
                {
                    long fileLength = reader.ReadInt64();
                    fullFilePath = Path.Combine(tempDirPath, _modelFileRepo + ".bin");
                    using (var fs = new FileStream(fullFilePath, FileMode.Create, FileAccess.Write))
                    {
                        long actualRead = reader.BaseStream.CopyRange(fs, fileLength);
                        Host.Assert(actualRead == fileLength);
                    }
                });
                Host.CheckDecode(load);

                _savedModelPath = fullFilePath;
                Module = TorchUtils.LoadTorchModel(Host, _savedModelPath).Module;
            }
            catch (Exception)
            {
                Directory.Delete(tempDirPath, true);
                throw;
            }
            Host.CheckDecode(_savedModelPath != null);
            Host.CheckDecode(Module != null);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // string output column name
            // int: number of input columns
            // for each input column
            //   string: input column name
            //   int: length of the inputshape array for this input column
            //   for each element in the inputshape
            //      long: element
            // stream: torch JIT module

            Host.AssertNonWhiteSpace(OutputColumnName);
            ctx.SaveNonEmptyString(OutputColumnName);

            Host.AssertNonEmpty(InputColumnNames);
            Host.AssertNonEmpty(InputShapes);
            Host.Assert(InputColumnNames.Length == InputShapes.Length);
            ctx.Writer.Write(InputColumnNames.Length);
            for (int i = 0; i < InputColumnNames.Length; i++)
            {
                ctx.SaveNonEmptyString(InputColumnNames[i]);
                Host.AssertNonEmpty(InputShapes[i]);
                ctx.Writer.Write(InputShapes[i].Length);
                foreach (var dim in InputShapes[i])
                    ctx.Writer.Write(dim);
            }

            // REVIEW: The below requires the model file not to have been moved.
            // A better alternative would be to use the Torch::JIT::Module::Save() method that uses a stream.
            Host.AssertNonWhiteSpace(_savedModelPath);
            ctx.SaveBinaryStream(_modelFileRepo, writer =>
            {
                using (var fs = new FileStream(_savedModelPath, FileMode.Open))
                {
                    long fileLength = fs.Length;
                    writer.Write(fileLength);
                    long actualWritten = fs.CopyRange(writer.BaseStream, fileLength);
                    Host.Assert(actualWritten == fileLength);
                }
            });
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, TorchScoringEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumnNames, nameof(options.InputColumnNames));
            env.CheckValue(options.OutputColumnName, nameof(options.OutputColumnName));

            return new TorchScoringEstimator(env, options).Fit(input).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => new TorchScoringTransformer(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new TorchScoringTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema)
            => new Mapper(this, inputSchema);

        ~TorchScoringTransformer()
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

        private sealed class Mapper : MapperBase
        {
            private readonly TorchScoringTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _needReshape;

            public Mapper(TorchScoringTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[_parent.InputColumnNames.Length];
                _needReshape = new bool[_parent.InputColumnNames.Length];

                for (int i = 0; i < _parent.InputColumnNames.Length; i++)
                {
                    // Check presence of input columns.
                    if (!inputSchema.TryGetColumnIndex(_parent.InputColumnNames[i], out _inputColIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "source", _parent.InputColumnNames[i]);

                    // Check that input columns are known-size vectors of Single.
                    var type = inputSchema[_inputColIndices[i]].Type;
                    if (!(type is VectorDataViewType vectorType) || !vectorType.IsKnownSize || vectorType.ItemType != NumberDataViewType.Single)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "score", _parent.InputColumnNames[i], "known-size vector of Single", type.ToString());

                    var colTypeDims = vectorType.Dimensions.Select(dim => (long)dim).ToArray();
                    var colShapeLength = colTypeDims.Length;
                    if (colShapeLength > 1)
                    {
                        if (colShapeLength != _parent.InputShapes[i].Length)
                            throw Host.Except($"Input shape mismatch: Input Column '{_parent.InputColumnNames[i]}' vector shape length {colShapeLength} does not match expected {_parent.InputShapes[i].Length}.");

                        for (int j = 0; j < colShapeLength; j++)
                        {
                            if (colTypeDims[j] != _parent.InputShapes[i][j])
                                throw Host.Except($"Input shape mismatch: Input Column '{_parent.InputColumnNames[i]}' dimension {j} of size {colTypeDims[j]} does not match expected size {_parent.InputShapes[j]}.");
                        }
                    }
                    else
                    {
                        // If the column in the input schema is one dimension we make sure that the total size of the Torch shape matches.
                        long valCount = _parent.InputShapes[i].Aggregate(1, (long x, long y) => x * y);

                        if (vectorType.Size != valCount)
                            throw Host.Except($"Input size {vectorType.Size} does not match with expect size {valCount}.");
                        _needReshape[i] = true;
                    }
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return new DataViewSchema.DetachedColumn[]
                {
                    // REVIEW: we need to deal with how we get the output size for the vector.
                    new DataViewSchema.DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single), null)
                };
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);

                ValueGetter<VBuffer<float>> valuegetter = (ref VBuffer<float> dst) =>
                {
                    var inputTensors = new ITorchTensor<float>[_inputColIndices.Length];

                    for (int i = 0; i < _inputColIndices.Length; i++)
                        inputTensors[i] = CreateTensorValueGetterVec<float>(input, _inputColIndices[i], _parent.InputShapes[i]);

                    ITorchTensor<float> result = _parent.Module.Forward(inputTensors);
                    var resultSize = result.NumberOfElements;
                    var editor = VBufferEditor.Create(ref dst, (int)resultSize);

                    result.Data.CopyTo(editor.Values);
                    dst = editor.Commit();

                    // Dispose the Torch tensors used to computer the dst vector.
                    if (inputTensors != null)
                        foreach (var tensor in inputTensors)
                            tensor?.Dispose();
                    result.Dispose();
                };
                return valuegetter;
            }

            private ITorchTensor<T> CreateTensorValueGetterVec<T>(DataViewRow input, int colIndex, long[] shape)
            {
                var srcgetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
                VBuffer<T> vBuffer = default;
                T[] denseData = default;
                ITorchTensor<T> tensor = default;

                // Get the input data.
                srcgetter(ref vBuffer);

                // _denseData.Length can be greater than _vBuffer.Length sometime after
                // Utils.EnsureSize is exectued. Use _vBuffer.Length to access the elements in _denseData.
                // This is done to reduce memory allocation every time tensor is created.
                Utils.EnsureSize(ref denseData, vBuffer.Length, keepOld: false);
                vBuffer.CopyTo(denseData);

                // Write it to a torch tensor.
                tensor = denseData.ToTorchTensor(shape);

                if (_needReshape[colIndex])
                {
                    var result = tensor.View(_parent.InputShapes[colIndex]);
                    tensor.Dispose(); // Need to dipose the partial tensor here otherwise we will have a memory leack
                    return result;
                }

                return tensor;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                // Range goes from 0 to 1 because there is only one output column.
                return col => Enumerable.Range(0, 1).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            private protected override void SaveModel(ModelSaveContext ctx)
                => _parent.SaveModel(ctx);
        }
    }

    /// <summary>
    /// Estimator for the <see cref="TorchScoringTransformer"/>.
    /// </summary>
    public sealed class TorchScoringEstimator : TrivialEstimator<TorchScoringTransformer>
    {
        /// <summary>
        /// The options for the <see cref="TorchScoringTransformer"/>.
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// The name of the output column of the transformation.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the output column", Name = "Name", ShortName = "name", SortOrder = 2)]
            public string OutputColumnName;

            /// <summary>
            /// The names of the columns containing the inputs for the model. If <see langword="null"/>, this defaults to <see cref="OutputColumnName"/>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The names of the input columns", Name = "Sources", ShortName = "src", SortOrder = 1)]
            public string[] InputColumnNames;

            /// <summary>
            /// The shape of the model input.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The shape of the input tensor", ShortName = "shape", SortOrder = 0)]
            public long[][] InputShapes;

            /// <summary>
            /// Location of the Torch model.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "Torch model used by the transform.", SortOrder = 0)]
            internal string ModelLocation = null;
        }

        private readonly string _outputColumnName;
        private readonly string[] _inputColumnNames;
        private readonly long[][] _inputShapes;

        internal TorchScoringEstimator(IHostEnvironment env, Options options, TorchModel torchModel)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchScoringEstimator)),
                  new TorchScoringTransformer(env, torchModel.Module, options.OutputColumnName, options.InputColumnNames, options.InputShapes, options.ModelLocation))
        {
            Host.CheckNonEmpty(options.OutputColumnName, nameof(options.OutputColumnName));
            Host.CheckValue(options.InputShapes, nameof(options.InputShapes));
            Host.CheckParam(!options.InputShapes.Any(x => x.Any(y => y <= 0)), nameof(options.InputShapes), "Unknown shape dimensions not supported.");
            _outputColumnName = options.OutputColumnName;
            _inputColumnNames = options.InputColumnNames ?? new[] { options.OutputColumnName };
            _inputShapes = options.InputShapes;
        }

        internal TorchScoringEstimator(IHostEnvironment env, Options options)
            : this(env, options, TorchUtils.LoadTorchModel(env, options.ModelLocation))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var resultDic = inputSchema.ToDictionary(x => x.Name);

            for (var i = 0; i < _inputColumnNames.Length; i++)
            {
                var input = _inputColumnNames[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (col.Kind != SchemaShape.Column.VectorKind.Vector || col.ItemType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector of Single", col.GetTypeString());
            }

            resultDic[_outputColumnName] = new SchemaShape.Column(
                _outputColumnName,
                SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single,
                false);

            return new SchemaShape(resultDic.Values);
        }
    }
}
