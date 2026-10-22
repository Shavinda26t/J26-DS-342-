import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  IconActivity,
  IconChevronLeft,
  IconHome,
  IconLayoutGrid,
  IconNetwork,
  IconRefreshCw,
  IconShield,
  IconShieldCheck,
} from '../common/Icons';

const NAV_SECTIONS = [
  {
    title: 'Overview',
    items: [{ to: '/dashboard', label: 'Dashboard', icon: IconLayoutGrid }],
  },
  {
    title: 'Research Modules',
    items: [
      { to: '/fleet', label: 'Fleet Intelligence', badge: 'C1', icon: IconActivity },
      { to: '/digital-twin', label: 'Digital Twin', badge: 'C2', icon: IconRefreshCw },
      { to: '/causal-xai', label: 'Causal XAI', badge: 'C3', icon: IconNetwork },
      { to: '/safe-rl', label: 'Safe Optimization', badge: 'C4', icon: IconShield },
    ],
  },
];

const DashboardSidebar = ({ collapsed = false, mobileOpen = false, onToggleCollapse, onNavigate }) => (
  <aside className={`ui-sidebar ${mobileOpen ? 'ui-sidebar--open' : ''}`.trim()} aria-label="Dashboard navigation">
    <div className="ui-sidebar__top">
      <NavLink to="/" className="ui-brand" onClick={onNavigate} title="HDD Health Framework">
        <span className="ui-brand__mark">
          <IconShieldCheck size={18} />
        </span>
        <span className="ui-brand__text">HDD Health</span>
      </NavLink>
      {onToggleCollapse && (
        <button
          type="button"
          className="ui-sidebar__collapse"
          onClick={onToggleCollapse}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <IconChevronLeft size={16} />
        </button>
      )}
    </div>

    <nav className="ui-sidebar__nav">
      {NAV_SECTIONS.map((section) => (
        <React.Fragment key={section.title}>
          <div className="ui-sidebar__section">{section.title}</div>
          {section.items.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `ui-sidebar__link ${isActive ? 'active' : ''}`.trim()}
                title={item.badge ? `${item.label} (${item.badge})` : item.label}
                onClick={onNavigate}
              >
                <Icon size={19} />
                <span className="ui-sidebar__label">{item.label}</span>
                {item.badge && <span className="ui-sidebar__badge">{item.badge}</span>}
              </NavLink>
            );
          })}
        </React.Fragment>
      ))}
    </nav>

    <div className="ui-sidebar__bottom">
      <NavLink to="/" className="ui-sidebar__link" title="Back to public site" onClick={onNavigate}>
        <IconHome size={19} />
        <span className="ui-sidebar__label">Public site</span>
      </NavLink>
    </div>
  </aside>
);

export default DashboardSidebar;
