import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import AuthLayout from './components/layout/AuthLayout';
import DashboardLayout from './components/layout/DashboardLayout';
import RequireAuth from './components/auth/RequireAuth';
import Home from './pages/Home';
import Login from './pages/Login';
import About from './pages/About';
import Contact from './pages/Contact';
import NotFound from './pages/NotFound';
import Dashboard from './pages/Dashboard';
import FleetIntelligence from './pages/FleetIntelligence';
import DigitalTwin from './pages/DigitalTwin';
import CausalXAI from './pages/CausalXAI';
import SafeOptimization from './pages/SafeOptimization';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<MainLayout />}>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="*" element={<NotFound />} />
        </Route>

        <Route element={<AuthLayout />}>
          <Route path="/login" element={<Login />} />
        </Route>

        <Route element={<RequireAuth />}>
          <Route element={<DashboardLayout />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/fleet" element={<FleetIntelligence />} />
            <Route path="/digital-twin" element={<DigitalTwin />} />
            <Route path="/causal-xai" element={<CausalXAI />} />
            <Route path="/safe-rl" element={<SafeOptimization />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
