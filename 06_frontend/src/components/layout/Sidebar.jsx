import React from 'react';
import { NavLink } from 'react-router-dom';

const Sidebar = () => (
  <aside className="sidebar">
    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#64748b', fontWeight: '700', padding: '0 14px 4px 14px' }}>
      Framework Modules
    </div>
    <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>
      Dashboard
    </NavLink>
    <NavLink to="/fleet" className={({ isActive }) => (isActive ? 'active' : '')}>
      Fleet Intelligence (C1)
    </NavLink>
    <NavLink to="/digital-twin" className={({ isActive }) => (isActive ? 'active' : '')}>
      Digital Twin (C2)
    </NavLink>
    <NavLink to="/causal-xai" className={({ isActive }) => (isActive ? 'active' : '')}>
      Causal XAI (C3)
    </NavLink>
    <NavLink to="/safe-rl" className={({ isActive }) => (isActive ? 'active' : '')}>
      Safe Optimization (C4)
    </NavLink>
  </aside>
);

export default Sidebar;
