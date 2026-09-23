import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './pages/Dashboard';
import FleetIntelligence from './pages/FleetIntelligence';
import DigitalTwin from './pages/DigitalTwin';
import CausalXAI from './pages/CausalXAI';
import SafeOptimization from './pages/SafeOptimization';

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Navbar />
        <div className="main-body">
          <Sidebar />
          <main className="content-area">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/fleet" element={<FleetIntelligence />} />
              <Route path="/digital-twin" element={<DigitalTwin />} />
              <Route path="/causal-xai" element={<CausalXAI />} />
              <Route path="/safe-rl" element={<SafeOptimization />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
