const { QdrantClient } = require('@qdrant/js-client-rest');
const fs = require('fs');
const path = require('path');

const logFile = path.join(__dirname, 'test_qdrant_output.log');
const logStream = fs.createWriteStream(logFile, { flags: 'w' });

function log(msg, obj) {
  const line = obj ? `${msg} ${JSON.stringify(obj, null, 2)}` : msg;
  console.log(line);
  logStream.write(line + '\n');
}

const url = 'https://88166896-11e5-431a-9ffb-1783722b352c.sa-east-1-0.aws.cloud.qdrant.io';
const apiKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODRmNWYxOWYtZjE1My00MzJjLTk3MDktYTg5MDE1MWFjNWE3In0.GBYuOI6fR8dfPykQh0KGBFCoDP71fqDkEf8rpeK3_rg';

const client = new QdrantClient({ url, apiKey, checkCompatibility: false });

async function run() {
  log("=== QDRANT DIAGNOSTIC RUN ===");
  
  try {
    log("Calling scroll with string type field_schema...");
    const res = await client.scroll('collection_gemini', {
      filter: {
        must: [{ key: 'customerId', match: { value: 'cust_12345' } }]
      },
      limit: 10,
      with_payload: true,
      with_vector: false
    });
    log("Scroll succeeded!", res);
  } catch (err) {
    log("Scroll failed!");
    log("Status: " + err.status);
    log("Message: " + err.message);
    log("Data details:", err.data);
    if (err.response) {
      try {
        const body = await err.response.text();
        log("Response body:", body);
      } catch (e) {
        log("Could not read response body: " + e.message);
      }
    }
  }

  try {
    log("Calling createPayloadIndex with string schema 'keyword'...");
    const res = await client.createPayloadIndex('collection_gemini', {
      field_name: 'customerId',
      field_schema: 'keyword'
    });
    log("Index creation (string schema) succeeded!", res);
  } catch (err) {
    log("Index creation (string schema) failed!");
    log("Status: " + err.status);
    log("Message: " + err.message);
    log("Data details:", err.data);
    if (err.response) {
      try {
        const body = await err.response.text();
        log("Response body:", body);
      } catch (e) {
        log("Could not read response body: " + e.message);
      }
    }
  }

  try {
    log("Calling createPayloadIndex with object schema { type: 'keyword' }...");
    const res = await client.createPayloadIndex('collection_gemini', {
      field_name: 'customerId',
      field_schema: { type: 'keyword' }
    });
    log("Index creation (object schema) succeeded!", res);
  } catch (err) {
    log("Index creation (object schema) failed!");
    log("Status: " + err.status);
    log("Message: " + err.message);
    log("Data details:", err.data);
  }

  logStream.end();
}

run();
