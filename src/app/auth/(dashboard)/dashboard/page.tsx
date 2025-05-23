'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';

const themes = {
  primary: '#4b6fff',
  background: '#f0f4ff',
  accent: '#263ca6',
};

export default function HRLangGraphUI() {
  const [messages, setMessages] = useState([]);  // start empty
  const [query, setQuery] = useState('');
  const [sessionId, setSessionId] = useState(() => uuidv4());
  const [isLoading, setIsLoading] = useState(false);
  const chatRef = useRef(null);

  // Delay the initial system message by 1 second
  useEffect(() => {
    const timer = setTimeout(() => {
      setMessages([{ role: 'system', text: 'I am ready to help you.' }]);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const handleSend = async () => {
    if (!query.trim()) return;

    const userMessage = { role: 'user', text: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);

    try {
      const response = await fetch(
        `http://localhost:5000/chat?chat=${encodeURIComponent(query)}&session_id=${sessionId}`
      );
      const data = await response.json();

      if (data?.user_messages) {
        const systemMessages = [{
          role: 'system',
          text: data.user_messages[data.user_messages.length - 1].replace(/^system:\s*/, '')
        }];
        setMessages((prev) => [...prev, ...systemMessages]);
      }
    } catch (error) {
      console.error('Error fetching response:', error);
      setMessages((prev) => [...prev, { role: 'system', text: 'Error fetching response from server.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    chatRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleNewSession = () => {
    const newId = uuidv4();
    const newTab = window.open(window.location.href, '_blank');
    newTab.onload = () => {
      newTab.sessionStorage.setItem('sessionId', newId);
    };
  };

  useEffect(() => {
    const storedId = sessionStorage.getItem('sessionId');
    if (storedId) setSessionId(storedId);
  }, []);

  return (
    <main className="min-h-screen w-full p-8" style={{ background: themes.background }}>
      <div className="flex justify-between items-center mb-4">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-2xl font-bold text-center"
          style={{ color: themes.accent }}
        >
          HR Chat Agent - Squareshift
        </motion.h1>
        <Button onClick={handleNewSession} className="ml-4">New Session</Button>
      </div>

      <Card className="w-full max-w-4xl mx-auto rounded-2xl shadow-xl">
        <CardContent className="p-6 space-y-4">
          <div className="h-[60vh] overflow-y-auto space-y-3">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`prose prose-sm p-3 rounded-lg w-fit max-w-[80%] text-sm whitespace-pre-wrap ${
                  msg.role === 'user' ? 'bg-blue-200 ml-auto' : 'bg-gray-100 mr-auto'
                }`}
              >
                <ReactMarkdown
                  components={{
                    ul: ({ children }) => <ul className="list-disc pl-6">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal pl-6">{children}</ol>,
                    li: ({ children }) => <li className="mb-1">{children}</li>,
                  }}
                >
                  {msg.text}
                </ReactMarkdown>
              </div>
            ))}
            {isLoading && <div className="text-gray-400 italic">Thinking...</div>}
            <div ref={chatRef} />
          </div>

          <div className="flex gap-2">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask your HR assistant..."
              disabled={isLoading}
              className="flex-grow p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              rows={2}
            />
            <Button
              style={{ backgroundColor: themes.primary }}
              onClick={handleSend}
              disabled={!query.trim() || isLoading}
            >
              Send
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
