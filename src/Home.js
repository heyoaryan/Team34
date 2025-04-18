// src/pages/Home.js
import React from 'react';
import './Home.css';
import { useNavigate } from 'react-router-dom';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

function Home() {
  const navigate = useNavigate();

  const handleStartChat = () => {
    navigate("/chat");
  };

  const chartData = {
    labels: ['Anxiety', 'Depression', 'Bipolar', 'Schizophrenia', 'Eating Disoder', 'PTSD'],
    datasets: [
      {
        label: 'India (%)',
        data: [30, 28, 5, 6, 15, 5],
        backgroundColor: 'rgba(106, 17, 203, 0.6)',
        borderRadius: 100,
      },
      {
        label: 'Global (%)',
        data: [31, 28, 6, 5, 10, 80],
        backgroundColor: 'rgba(37, 117, 252, 0.6)',
        borderRadius: 100,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#333', font: { family: 'Poppins' } }
      }
    },
    scales: {
      x: {
        ticks: { color: '#333', font: { family: 'Poppins' } },
        grid: { display: false }
      },
      y: {
        beginAtZero: true,
        ticks: { color: '#333', font: { family: 'Poppins' } },
        grid: { borderDash: [4, 4] }
      }
    }
  };

  return (
    <div className="home-container">
      <header className="navbar">
        <h1 className="logo">🧠 MindMitra</h1>
        <button className="login-btn" onClick={() => navigate('/login')}>Login</button>
      </header>

      <main className="main-section video-hero">
        <video autoPlay loop muted playsInline className="hero-video">
          <source src="/Video/Home.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>

        <div className="video-overlay">
          <h2 className="main-heading">Welcome To MindMitra...</h2>
          <p>Your AI-powered companion for emotional well-being.</p>
          <button className="start-chat-btn" onClick={handleStartChat}>
            Start Chat
          </button>
        </div>
      </main>

      <section className="info-section fadeInUp">
        <h3>About MindMitra</h3>
        <p>
          MindMitra is an AI-based mental health chatbot that provides a safe, anonymous, and emotionally intelligent space for individuals to express, reflect, and heal — anytime, anywhere.
        </p>
      </section>

      <section className="info-section fadeInUp features-section">
        <h3>Key Features</h3>
        <div className="feature-grid">
          <div className="feature-card">⏰ 24/7 Support</div>
          <div className="feature-card">🎤 Voice-Enabled Chatting</div>
          <div className="feature-card">🌐 Multilingual Conversations</div>
          <div className="feature-card">💬 Emotion-Based Responses</div>
          <div className="feature-card">📊 Wellness Summary</div>
          <div className="feature-card">🧠 Sentiment Analysis</div>
          <div className="feature-card">🔐 Anonymous & Secure</div>
          <div className="feature-card">💾 Chat History Tracking</div>
          <div className="feature-card">📥 Personalized Suggestions</div>
          <div className="feature-card">🧘 Relaxation Recommendations</div>
        </div>
      </section>

      <section className="info-section fadeInUp">
        <h3>How It Works</h3>
        <p>
          You simply start chatting — MindMitra uses natural language AI, detects your tone, understands your mood, and guides you in a calm, secure environment. You can talk or type, in multiple languages.
        </p>
      </section>

      <section className="info-section fadeInUp">
        <h3>📊 Mental Disorder Statistics: India vs Global</h3>
        <p>This chart compares the prevalence of common mental disorders between India and the global average (percent of population).</p>
        <div style={{ maxWidth: '700px', margin: '0 auto' }}>
          <Bar data={chartData} options={chartOptions} />
        </div>
      </section>

      <section className="info-section fadeInUp">
        <h3>🧠 Why Mental Health Matters - WHO Insights</h3>
        <p>
          According to the <strong>World Health Organization (WHO)</strong>, over <strong>970 million people globally</strong> suffer from mental or behavioral disorders.
          India alone faces over <strong>200 million mental health cases</strong> annually — and access to therapists is low. MindMitra aims to bridge that gap with emotional AI, voice support, and total privacy.
        </p>
      </section>

      <footer className="footer fadeInUp">
        © 2025 MindMitra — Empowering Mental Health with AI 💜
      </footer>
    </div>
  );
}

export default Home;
