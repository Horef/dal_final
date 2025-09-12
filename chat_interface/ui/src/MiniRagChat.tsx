import React, { useState } from 'react'

export default function MiniRagChat() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  async function sendMessage() {
    if (!input.trim()) return
    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessage.content,
          history: messages,
          top_k: 4,
        }),
      })
      const data = await res.json()
      const botMessage = { role: 'assistant', content: data.answer || 'No answer.' }
      setMessages(prev => [...prev, botMessage])
    } catch (err) {
      console.error(err)
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error contacting API.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen p-4">
      <div className="flex-1 overflow-y-auto border rounded p-2 mb-2 bg-gray-50">
        {messages.map((m, i) => (
          <div key={i} className={`mb-2 ${m.role === 'user' ? 'text-blue-600' : 'text-green-700'}`}>
            <strong>{m.role}:</strong> {m.content}
          </div>
        ))}
        {loading && <div className="text-gray-500">…thinking</div>}
      </div>

      <div className="flex gap-2">
        <input
          className="flex-1 border rounded p-2"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Ask MiniRAG…"
        />
        <button
          className="px-4 py-2 bg-black text-white rounded"
          onClick={sendMessage}
          disabled={loading}
        >
          Send
        </button>
      </div>
    </div>
  )
}
