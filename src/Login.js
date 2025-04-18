// src/pages/Login.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Auth.css';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const res = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await res.json();

      if (data.success) {
        localStorage.setItem('username', email);
        window.dispatchEvent(new Event('storage'));
        alert('Login successful!');
        navigate('/chat');
      } else {
        alert(data.message || 'Login failed.');
      }
    } catch (err) {
      console.error('Login error:', err);
      alert('Something went wrong while logging in.');
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <h2>Login to MindMitra</h2>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button onClick={handleLogin}>Login</button>
        <p>
          Don’t have an account?
          <span className="auth-link" onClick={() => navigate('/signup')}>
            Sign up
          </span>
        </p>
      </div>
    </div>
  );
}

export default Login;
