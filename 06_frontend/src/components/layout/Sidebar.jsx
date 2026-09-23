import React from 'react';
import { Link } from 'react-router-dom';
const Sidebar = () => (
  <aside className="sidebar">
    <Link to="/">Dashboard</Link>
    <Link to="/causal-xai">Causal XAI</Link>
  </aside>
);
export default Sidebar;
