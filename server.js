// server.js - OpenAI to NVIDIA NIM Proxy (fix de errores en stream + mensajes vacíos)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'nvidia/nemotron-3-nano-30b-a3b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-sonnet': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gemini-pro': 'nvidia/nemotron-3-nano-30b-a3b'
};

// Lee el body de error aunque axios esté en modo stream
async function readErrorBody(error) {
  const res = error.response;
  if (!res) return { message: error.message || 'Request failed' };

  let data = res.data;

  // Si es un stream, consumirlo
  if (data && typeof data.pipe === 'function') {
    try {
      const chunks = [];
      for await (const chunk of data) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      const text = Buffer.concat(chunks).toString('utf8');
      try {
        return JSON.parse(text);
      } catch {
        return { message: text || error.message };
      }
    } catch (e) {
      return { message: error.message || 'Failed to read error body' };
    }
  }

  if (typeof data === 'string') {
    try {
      return JSON.parse(data);
    } catch {
      return { message: data };
    }
  }

  if (data && typeof data === 'object') {
    // Evitar meter sockets u objetos circulares
    try {
      JSON.stringify(data);
      return data;
    } catch {
      return { message: data.message || data.error?.message || error.message };
    }
  }

  return { message: error.message || 'Request failed' };
}

// Quitar mensajes con content vacío (causa típica del 400 de NIM)
function cleanMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter(m => m && typeof m === 'object')
    .map(m => {
      const role = m.role || 'user';
      let content = m.content;

      if (content == null) content = '';
      if (typeof content !== 'string') {
        // Si viene como array (multimodal), convertir a texto simple
        if (Array.isArray(content)) {
          content = content
            .map(part => (typeof part === 'string' ? part : part?.text || ''))
            .join('');
        } else {
          content = String(content);
        }
      }

      return { role, content };
    })
    .filter(m => m.content.trim().length > 0 || m.role === 'system');
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'NIM Proxy', streaming: true });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    let nimModel = MODEL_MAPPING[model] || model;
    const cleanedMessages = cleanMessages(messages);

    if (cleanedMessages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'No valid messages after cleaning empty content',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages,
      temperature: temperature ?? 0.8,
      max_tokens: max_tokens || 2048,
      stream: !!stream
    };

    if (nimRequest.temperature > 1.0) nimRequest.temperature = 1.0;
    if (nimRequest.temperature < 0) nimRequest.temperature = 0;

    if (nimRequest.max_tokens > 4096) nimRequest.max_tokens = 4096;

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    // ===== STREAMING =====
    if (stream) {
      const response = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json',
            Accept: 'text/event-stream'
          },
          responseType: 'stream',
          timeout: 300000,
          validateStatus: () => true // para poder leer body si status >= 400
        }
      );

      if (response.status >= 400) {
        const chunks = [];
        for await (const chunk of response.data) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        }
        const text = Buffer.concat(chunks).toString('utf8');
        let parsed;
        try {
          parsed = JSON.parse(text);
        } catch {
          parsed = { message: text };
        }
        console.error('=== NIM ERROR START ===');
        console.error('Status:', response.status);
        console.error('Response data:', JSON.stringify(parsed, null, 2));
        console.error('=== NIM ERROR END ===');

        return res.status(response.status).json({
          error: {
            message: parsed?.message || parsed?.error?.message || text || 'Request failed',
            type: 'invalid_request_error',
            code: response.status,
            details: parsed
          }
        });
      }

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      response.data.on('data', (chunk) => {
        if (!res.writableEnded) res.write(chunk);
      });
      response.data.on('end', () => {
        if (!res.writableEnded) res.end();
      });
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        if (!res.writableEnded) res.end();
      });
      req.on('close', () => {
        try { response.data.destroy(); } catch {}
      });
      return;
    }

    // ===== NO STREAMING =====
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: 300000
      }
    );

    const data = response.data;
    const choice = data.choices?.[0];
    let content = choice?.message?.content || '';

    if (SHOW_REASONING && choice?.message?.reasoning_content) {
      content = `<think>\n\( {choice.message.reasoning_content}\n</think>\n\n \){content}`;
    }

    res.json({
      id: data.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: data.created || Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content },
        finish_reason: choice?.finish_reason || 'stop'
      }],
      usage: data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });

  } catch (error) {
    const status = error.response?.status || 500;
    const detail = await readErrorBody(error);

    console.error('=== NIM ERROR START ===');
    console.error('Status:', status);
    console.error('Message:', error.message);
    try {
      console.error('Response data:', JSON.stringify(detail, null, 2));
    } catch {
      console.error('Response data:', detail?.message || String(detail));
    }
    console.error('=== NIM ERROR END ===');

    const message =
      detail?.message ||
      detail?.error?.message ||
      detail?.detail ||
      error.message ||
      'Request failed';

    if (!res.headersSent) {
      res.status(status).json({
        error: {
          message: typeof message === 'string' ? message : JSON.stringify(message),
          type: 'invalid_request_error',
          code: status,
          details: detail
        }
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`NIM Proxy running on port ${PORT}`);
});
