// src/components/Dashboard.js
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Chart from 'chart.js/auto';
import './Dashboard.css';

let barChartInstance = null;
let pieChartInstance = null;

function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);

  useEffect(() => {
    const user = localStorage.getItem('username');
    if (!user) {
      navigate('/login');
    } else {
      fetch(`http://localhost:5000/stats?username=${user}`)
        .then(res => res.json())
        .then(data => {
          setStats(data);
          setTimeout(() => renderCharts(data.emotionCount), 100);
        });
    }
  }, [navigate]);

  const renderCharts = (emotionData) => {
    const barCanvas = document.getElementById("barChart");
    const pieCanvas = document.getElementById("pieChart");

    if (barChartInstance) {
      barChartInstance.destroy();
    }
    if (pieChartInstance) {
      pieChartInstance.destroy();
    }

    if (barCanvas) {
      barChartInstance = new Chart(barCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: Object.keys(emotionData),
          datasets: [{
            label: "Emotions",
            data: Object.values(emotionData),
            backgroundColor: ["#4caf50", "#2196f3", "#ff9800", "#e91e63", "#9c27b0", "#00bcd4"]
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
        }
      });
    }

    if (pieCanvas) {
      pieChartInstance = new Chart(pieCanvas.getContext("2d"), {
        type: "pie",
        data: {
          labels: Object.keys(emotionData),
          datasets: [{
            data: Object.values(emotionData),
            backgroundColor: ["#4caf50", "#2196f3", "#ff9800", "#e91e63", "#9c27b0", "#00bcd4"]
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
        }
      });
    }
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>🧠 MindMitra Dashboard</h1>
      </div>

      {stats ? (
        <>
          <div className="stats-cards">
            <div className="card">
              <h3>Total Sessions</h3>
              <p>{stats.sessions}</p>
            </div>
            <div className="card">
              <h3>Total Messages</h3>
              <p>{stats.totalMessages}</p>
            </div>
          </div>

          <div className="chart-box">
            <h3>Emotion Distribution (Bar Graph)</h3>
            <div className="chart-wrapper">
              <canvas id="barChart"></canvas>
            </div>
          </div>

          <div className="chart-box">
            <h3>Emotion Distribution (Pie Chart)</h3>
            <div className="chart-wrapper">
              <canvas id="pieChart"></canvas>
            </div>
          </div>
        </>
      ) : (
        <p className="loading">Loading your stats...</p>
      )}
    </div>
  );
}

export default Dashboard;
