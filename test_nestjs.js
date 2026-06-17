const axios = require('axios');

async function run() {
  try {
    console.log("Sending analytics query to NestJS...");
    const res = await axios.post('http://localhost:3000/rag/projects/analytics', {
      customerId: 'cust_12345',
      metric: 'estimatedValue'
    });
    console.log("Response:", res.data);
  } catch (err) {
    console.error("Error from NestJS:");
    console.error(err);
  }
}

run();
