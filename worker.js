export default {
  async fetch(request, env) {
    // Handle CORS preflight requests
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    try {
      const result = await env.MY_DB.prepare(
        "SELECT Id, CustomerName, OrderDate FROM [Order] ORDER BY ShippedDate DESC LIMIT 100",
      ).run();

      return new Response(JSON.stringify(result.results), {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*", // Allow all origins for simplicity
        },
      });
    } catch (e) {
      return new Response(JSON.stringify({ error: e.message }), {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }
  }
}
