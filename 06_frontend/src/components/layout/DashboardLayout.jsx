import React, { useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import DashboardSidebar from './DashboardSidebar';
import ProfileMenu from '../common/ProfileMenu';
import { IconMenu } from '../common/Icons';

const COLLAPSE_KEY = 'hddui.sidebar.collapsed';

const PAGE_META = {
  '/dashboard': { title: 'System Overview', subtitle: 'Fleet health, drift and reliability at a glance' },
  '/fleet': { title: 'Fleet Intelligence', subtitle: 'Behavioural drift & GenAI monitoring (C1)' },
  '/digital-twin': { title: 'Digital Twin', subtitle: 'Simulation & scenario modelling (C2)' },
  '/causal-xai': { title: 'Causal XAI', subtitle: 'Root-cause explanations & evidence (C3)' },
  '/safe-rl': { title: 'Safe Optimization', subtitle: 'Risk-aware migration policies (C4)' },
};

const DashboardLayout = () => {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === '1';
    } catch {
      return false;
    }
  });
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  const toggleCollapse = () => {
    setCollapsed((value) => {
      const next = !value;
      try {
        localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0');
      } catch {
        /* storage unavailable */
      }
      return next;
    });
  };

  const meta = PAGE_META[location.pathname] || PAGE_META['/dashboard'];

  return (
    <div className={`ui-dash ${collapsed ? 'ui-dash--collapsed' : ''}`.trim()} data-ui-theme="dark">
      <DashboardSidebar
        collapsed={collapsed}
        mobileOpen={mobileOpen}
        onToggleCollapse={toggleCollapse}
        onNavigate={() => setMobileOpen(false)}
      />
      {mobileOpen && <div className="ui-dash__overlay" onClick={() => setMobileOpen(false)} aria-hidden="true" />}
      <div className="ui-dash__main">
        <header className="ui-dash__header">
          <button
            type="button"
            className="ui-icon-btn ui-dash__burger"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
          >
            <IconMenu size={18} />
          </button>
          <div className="ui-dash__heading">
            <div className="ui-dash__title">{meta.title}</div>
            <div className="ui-dash__subtitle">{meta.subtitle}</div>
          </div>
          <ProfileMenu />
        </header>
        <div className="ui-dash__content">
          <Outlet />
        </div>
      </div>
    </div>
  );
};

export default DashboardLayout;
