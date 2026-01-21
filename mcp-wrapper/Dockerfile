FROM node:20-slim

# Install curl for healthchecks (Coolify requires it)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all dependencies (including dev for build)
COPY package.json ./
RUN npm install

# Copy source code
COPY tsconfig.json ./
COPY src/ ./src/

# Build TypeScript
RUN npm run build

# Remove dev dependencies for smaller image
RUN npm prune --production

# Use node user (already exists in node image)
RUN chown -R node:node /app
USER node

# Expose port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:8766/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Start server
CMD ["npm", "start"]
