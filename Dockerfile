# Rich Learning — Docker Setup
#
# Two services:
#   1. rich-learning: The .NET 10 app (builds from source)
#   2. neo4j: Optional graph database backend
#
# Usage:
#   docker compose up              # Start with Neo4j backend
#   docker compose up rich-learning  # Just the app (uses LiteDB)

# ─── Stage 1: Build ───
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY rich-learning.sln ./
COPY src/RichLearning/RichLearning.csproj src/RichLearning/
RUN dotnet restore

COPY . .
WORKDIR /src/src/RichLearning
RUN dotnet publish -c Release -o /app --no-restore

# ─── Stage 2: Runtime ───
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime
WORKDIR /app

COPY --from=build /app .

# Default: run Split-MNIST demo with LiteDB (zero setup)
ENTRYPOINT ["dotnet", "RichLearning.dll"]
CMD ["SplitMnist", "--litedb"]
