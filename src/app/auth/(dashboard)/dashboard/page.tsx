// A React UI for LangGraph-powered HR Chat Agent
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';

const themes = {
  primary: '#4b6fff',
  background: '#f0f4ff',
  accent: '#263ca6',
};

export default function HRLangGraphUI() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const chatRef = useRef(null);

  const handleSend = () => {
    if (!query.trim()) return;
    const newMessages = [...messages, { role: 'user', text: query }];
    setMessages([...newMessages, { role: 'assistant', text: ` ${query}` }]); // Placeholder
    setQuery('');
  };

  useEffect(() => {
    chatRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <main className="min-h-screen w-full p-8" style={{ background: themes.background }}>
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-4xl font-bold mb-6 text-center"
        style={{ color: themes.accent }}
      >
        HR Chat Agent - Squyarelft
      </motion.h1>

      <Card className="w-full max-w-4xl mx-auto rounded-2xl shadow-xl">
        <CardContent className="p-6 space-y-4">
          <div className="h-[60vh] overflow-y-auto space-y-3">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg w-fit max-w-[80%] ${
                  msg.role === 'user' ? 'bg-blue-200 ml-auto' : 'bg-gray-100 mr-auto'
                }`}
              >
                {msg.text}
              </div>
            ))}
            <div ref={chatRef} />
          </div>

          <div className="flex gap-2">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask your HR assistant..."
              className="flex-grow p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={2}
            />
            <Button style={{ backgroundColor: themes.primary }} onClick={handleSend}>
              Send
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}