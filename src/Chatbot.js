import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './Chatbot.css';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [language, setLanguage] = useState('en');
  const [chatSessions, setChatSessions] = useState([]);
  const [currentSessionIndex, setCurrentSessionIndex] = useState(0);
  const [typing, setTyping] = useState(false);
  const chatEndRef = useRef(null);
  const navigate = useNavigate();

  const userEmail = localStorage.getItem("username") || null;

  useEffect(() => {
    const saved = localStorage.getItem('mindmitraChats');
    if (saved) {
      const parsed = JSON.parse(saved);
      setChatSessions(parsed);
      setMessages(parsed[0]?.chat || []);
    } else {
      const initial = [{ date: new Date().toLocaleString(), chat: [] }];
      setChatSessions(initial);
      setMessages([]);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('mindmitraChats', JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = { text: input, sender: 'user' };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput('');
    setTyping(true);

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          lang: language,
          username: userEmail || 'Guest'
        })
      });

      const data = await response.json();
      const botMsg = {
        text: data.reply,
        sender: 'bot',
        emotion: data.emotion,
        suggestions: data.suggestions
      };

      const finalMsgs = [...updatedMessages, botMsg];
      setMessages(finalMsgs);
      setTyping(false);

      const newSessions = [...chatSessions];
      if (!newSessions[currentSessionIndex]) {
        newSessions[currentSessionIndex] = {
          date: new Date().toLocaleString(),
          chat: []
        };
      }
      newSessions[currentSessionIndex].chat = finalMsgs;
      setChatSessions(newSessions);

      const utter = new SpeechSynthesisUtterance(data.reply);
      utter.lang = language;
      speechSynthesis.speak(utter);
    } catch (err) {
      console.error("Error:", err);
      setTyping(false);
    }
  };

  const handleVoiceInput = () => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = language;
    recognition.start();
    recognition.onresult = (event) => {
      const voiceText = event.results[0][0].transcript;
      setInput(voiceText);
    };
  };

  const startNewChat = () => {
    const newSession = { date: new Date().toLocaleString(), chat: [] };
    const updatedSessions = [...chatSessions, newSession];
    setChatSessions(updatedSessions);
    setCurrentSessionIndex(updatedSessions.length - 1);
    setMessages([]);
  };

  const clearChat = () => {
    const cleared = [...chatSessions];
    if (!cleared[currentSessionIndex]) {
      cleared[currentSessionIndex] = {
        date: new Date().toLocaleString(),
        chat: []
      };
    } else {
      cleared[currentSessionIndex].chat = [];
    }
    setMessages([]);
    setChatSessions(cleared);
  };

  const loadChat = (index) => {
    setMessages(chatSessions[index]?.chat || []);
    setCurrentSessionIndex(index);
  };

  const handleLogout = () => {
    localStorage.clear();
    window.location.href = '/';
  };

  const handleLogin = () => {
    window.location.href = '/login';
  };

  const handleDashboardClick = () => {
    if (!userEmail) {
      alert("Please log in to access the Analytical Dashboard.");
    } else {
      navigate('/dashboard');
    }
  };

  const handleCommunityClick = () => {
    alert("Community feature coming soon!");
  };

  const groupSessions = () => {
    const today = [];
    const yesterday = [];
    const past30 = [];

    const now = new Date();
    chatSessions.forEach((session, idx) => {
      const date = new Date(session.date);
      const diff = (now - date) / (1000 * 3600 * 24);

      if (diff < 1) today.push({ ...session, idx });
      else if (diff < 2) yesterday.push({ ...session, idx });
      else if (diff <= 30) past30.push({ ...session, idx });
    });

    return { today, yesterday, past30 };
  };

  const { today, yesterday, past30 } = groupSessions();

  const renderChatGroup = (label, group) => (
    group.length > 0 && (
      <>
        <h5 style={{ marginTop: '1rem', color: '#555' }}>{label}</h5>
        {group.map(({ idx, date, chat }) => (
          <div key={idx} className="history-card" onClick={() => loadChat(idx)}>
            <strong>{new Date(date).toLocaleString()}</strong>
            <p>{chat[0]?.text.slice(0, 40) || 'No messages yet'}...</p>
          </div>
        ))}
      </>
    )
  );

  return (
    <div className="chatbot-wrapper">
      <div className="chatbot-header">
        <div className="chatbot-logo">🧠 MindMitra</div>
        <div className="chatbot-controls">
          {userEmail ? (
            <>
              <span className="username-tag">👤 {userEmail}</span>
              <button onClick={handleLogout}>Logout</button>
            </>
          ) : (
            <button onClick={handleLogin}>Login</button>
          )}
        </div>
      </div>

      <div className="chatbot-container">
        <div className="sidebar">
          <h4>New Chat</h4>
          <button onClick={startNewChat}>➕ Start New Chat</button>

          <h4>Chat History</h4>
          {renderChatGroup("Today", today)}
          {renderChatGroup("Yesterday", yesterday)}
          {renderChatGroup("Past 30 Days", past30)}

          <h4>Options</h4>
          <button onClick={clearChat} className="clear-btn">🗑️ Clear Chat</button>

          <div className="features-block">
            <h4>Features</h4>
            <button onClick={handleDashboardClick}>📊 Analytical Dashboard</button>
            <button onClick={handleCommunityClick}>🌍 Community</button>
          </div>
        </div>

        <div
          className="chat-area"
          style={{
            backgroundImage: "url('/image/chat-area background.png')",
            backgroundRepeat: "no-repeat",
            backgroundPosition: "center center",
            backgroundSize: "cover",
          }}
        >
          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`msg ${msg.sender} ${msg.emotion}`}>
                {msg.text}
                {msg.suggestions?.length > 0 && (
                  <div className="suggestions">
                    {msg.suggestions.map((s, j) => (
                      <div key={j} className="suggestion">{s}</div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {typing && <div className="msg bot typing">Typing...</div>}
            <div ref={chatEndRef}></div>
          </div>

          <div className="input-box">
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
            </select>
            <input
              type="text"
              placeholder="Type or speak..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button onClick={handleSend}>Send</button>
            <button onClick={handleVoiceInput}>🎤</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;
