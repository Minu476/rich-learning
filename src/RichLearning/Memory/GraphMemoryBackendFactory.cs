using System.Reflection;
using Microsoft.Extensions.Logging;
using RichLearning.Abstractions;

namespace RichLearning.Memory;

public sealed record GraphMemoryBackendOptions
{
    public string Kind { get; init; } = "litedb";
    public string? LiteDbPath { get; init; }
    public string Neo4jUri { get; init; } = "bolt://localhost:7687";
    public string Neo4jUser { get; init; } = "neo4j";
    public string Neo4jPassword { get; init; } = "password";
    public string? Neo4jDatabase { get; init; }
    public string? CustomAssemblyPath { get; init; }
    public string? CustomTypeName { get; init; }
    public IReadOnlyDictionary<string, string> CustomSettings { get; init; }
        = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
}

public static class GraphMemoryBackendFactory
{
    public static GraphMemoryBackendOptions FromArgs(
        string[] args,
        string? defaultLiteDbPath = null,
        string? forcedKind = null)
    {
        var kind = forcedKind
            ?? GetArgumentValue(args, "--backend")
            ?? Environment.GetEnvironmentVariable("RICHLEARNING_BACKEND")
            ?? (args.Any(arg => arg.Equals("--litedb", StringComparison.OrdinalIgnoreCase))
                ? "litedb"
                : "litedb");

        return new GraphMemoryBackendOptions
        {
            Kind = kind,
            LiteDbPath = GetArgumentValue(args, "--db")
                ?? Environment.GetEnvironmentVariable("RICHLEARNING_LITEDB_PATH")
                ?? defaultLiteDbPath,
            Neo4jUri = Environment.GetEnvironmentVariable("NEO4J_URI") ?? "bolt://localhost:7687",
            Neo4jUser = Environment.GetEnvironmentVariable("NEO4J_USER") ?? "neo4j",
            Neo4jPassword = Environment.GetEnvironmentVariable("NEO4J_PASSWORD") ?? "password",
            Neo4jDatabase = GetArgumentValue(args, "--neo4j-database")
                ?? Environment.GetEnvironmentVariable("NEO4J_DATABASE"),
            CustomAssemblyPath = GetArgumentValue(args, "--backend-assembly")
                ?? Environment.GetEnvironmentVariable("RICHLEARNING_BACKEND_ASSEMBLY"),
            CustomTypeName = GetArgumentValue(args, "--backend-type")
                ?? Environment.GetEnvironmentVariable("RICHLEARNING_BACKEND_TYPE"),
            CustomSettings = ParseSettings(args)
        };
    }

    public static async Task<IGraphMemory> CreateAsync(
        GraphMemoryBackendOptions options,
        ILoggerFactory loggerFactory)
    {
        switch (options.Kind.Trim().ToLowerInvariant())
        {
            case "litedb":
                if (string.IsNullOrWhiteSpace(options.LiteDbPath))
                    throw new InvalidOperationException(
                        "LiteDB backend selected but no database path was provided. Use --db or RICHLEARNING_LITEDB_PATH.");

                return new LiteDbGraphMemory(
                    options.LiteDbPath,
                    loggerFactory.CreateLogger<LiteDbGraphMemory>());

            case "neo4j":
                return new Neo4jGraphMemory(
                    options.Neo4jUri,
                    options.Neo4jUser,
                    options.Neo4jPassword,
                    loggerFactory.CreateLogger<Neo4jGraphMemory>(),
                    options.Neo4jDatabase);

            case "custom":
                return await CreateCustomAsync(options, loggerFactory);

            default:
                throw new InvalidOperationException(
                    $"Unknown backend '{options.Kind}'. Supported values: litedb, neo4j, custom.");
        }
    }

    public static string Describe(GraphMemoryBackendOptions options) =>
        options.Kind.Trim().ToLowerInvariant() switch
        {
            "litedb" => $"LiteDB → {options.LiteDbPath}",
            "neo4j" => string.IsNullOrWhiteSpace(options.Neo4jDatabase)
                ? $"Neo4j → {options.Neo4jUri} (server default database)"
                : $"Neo4j → {options.Neo4jUri} / {options.Neo4jDatabase}",
            "custom" => $"Custom → {options.CustomTypeName}",
            _ => options.Kind
        };

    private static async Task<IGraphMemory> CreateCustomAsync(
        GraphMemoryBackendOptions options,
        ILoggerFactory loggerFactory)
    {
        if (string.IsNullOrWhiteSpace(options.CustomTypeName))
        {
            throw new InvalidOperationException(
                "Custom backend selected but no type was provided. Use --backend-type or RICHLEARNING_BACKEND_TYPE.");
        }

        var backendType = ResolveBackendType(options.CustomTypeName, options.CustomAssemblyPath);
        if (!typeof(IGraphMemory).IsAssignableFrom(backendType))
        {
            throw new InvalidOperationException(
                $"Type '{backendType.FullName}' does not implement {nameof(IGraphMemory)}.");
        }

        var settings = options.CustomSettings;

        var asyncFactory = backendType.GetMethod(
            "CreateAsync",
            BindingFlags.Public | BindingFlags.Static,
            [typeof(ILoggerFactory), typeof(IReadOnlyDictionary<string, string>)]);
        if (asyncFactory is not null)
        {
            var created = asyncFactory.Invoke(null, [loggerFactory, settings]);
            if (created is Task<IGraphMemory> task)
                return await task;
        }

        var syncFactory = backendType.GetMethod(
            "Create",
            BindingFlags.Public | BindingFlags.Static,
            [typeof(ILoggerFactory), typeof(IReadOnlyDictionary<string, string>)]);
        if (syncFactory?.Invoke(null, [loggerFactory, settings]) is IGraphMemory createdByFactory)
            return createdByFactory;

        var ctors = new Func<object?[]?, IGraphMemory?>[]
        {
            args => backendType.GetConstructor([typeof(ILoggerFactory), typeof(IReadOnlyDictionary<string, string>)])?.Invoke(args) as IGraphMemory,
            args => backendType.GetConstructor([typeof(IReadOnlyDictionary<string, string>)])?.Invoke(args) as IGraphMemory,
            args => backendType.GetConstructor([typeof(ILoggerFactory)])?.Invoke(args) as IGraphMemory,
            args => backendType.GetConstructor(Type.EmptyTypes)?.Invoke(args) as IGraphMemory
        };

        var candidateArgs = new object?[][]
        {
            [loggerFactory, settings],
            [settings],
            [loggerFactory],
            []
        };

        for (int i = 0; i < ctors.Length; i++)
        {
            var instance = ctors[i](candidateArgs[i]);
            if (instance is not null)
                return instance;
        }

        throw new InvalidOperationException(
            $"Unable to create custom backend '{backendType.FullName}'. Supported signatures are Create/CreateAsync(ILoggerFactory, IReadOnlyDictionary<string,string>) or constructors accepting ILoggerFactory and/or settings.");
    }

    private static Type ResolveBackendType(string typeName, string? assemblyPath)
    {
        if (!string.IsNullOrWhiteSpace(assemblyPath))
        {
            var assembly = Assembly.LoadFrom(Path.GetFullPath(assemblyPath));
            var loadedType = assembly.GetType(typeName, throwOnError: false, ignoreCase: false);
            if (loadedType is not null)
                return loadedType;
        }

        var exactType = Type.GetType(typeName, throwOnError: false, ignoreCase: false);
        if (exactType is not null)
            return exactType;

        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            var discovered = assembly.GetType(typeName, throwOnError: false, ignoreCase: false);
            if (discovered is not null)
                return discovered;
        }

        throw new InvalidOperationException(
            $"Could not resolve custom backend type '{typeName}'. If it is in an external assembly, provide --backend-assembly.");
    }

    private static string? GetArgumentValue(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i].Equals(name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }

        return null;
    }

    private static IReadOnlyDictionary<string, string> ParseSettings(string[] args)
    {
        var settings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < args.Length - 1; i++)
        {
            if (!args[i].Equals("--backend-setting", StringComparison.OrdinalIgnoreCase))
                continue;

            var raw = args[i + 1];
            var separator = raw.IndexOf('=');
            if (separator <= 0 || separator == raw.Length - 1)
                continue;

            settings[raw[..separator]] = raw[(separator + 1)..];
        }

        return settings;
    }
}