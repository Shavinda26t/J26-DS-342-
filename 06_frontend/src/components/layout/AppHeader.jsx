import React, { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { useTheme } from '../../hooks/useTheme';
import Button from '../common/Button';
import ProfileMenu from '../common/ProfileMenu';
import { IconMenu, IconMoon, IconShieldCheck, IconSun, IconX } from '../common/Icons';

const NAV_LINKS = [
  { to: '/', label: 'Home', end: true },
  { to: '/about', label: 'About' },
  { to: '/contact', label: 'Contact' },
];

const AppHeader = () => {
  const { isAuthenticated } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);

  const links = isAuthenticated
    ? [...NAV_LINKS, { to: '/dashboard', label: 'Dashboard' }]
    : NAV_LINKS;

  return (
    <header className="ui-header">
      <div className="ui-header__inner">
        <Link to="/" className="ui-brand" onClick={() => setMenuOpen(false)}>
          <span className="ui-brand__mark">
            <IconShieldCheck size={18} />
          </span>
          <span className="ui-brand__text">HDD Health Framework</span>
        </Link>

        <nav className={`ui-nav ${menuOpen ? 'ui-nav--open' : ''}`} aria-label="Main navigation">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => `ui-nav__link ${isActive ? 'active' : ''}`.trim()}
              onClick={() => setMenuOpen(false)}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="ui-header__actions">
          <button
            type="button"
            className="ui-icon-btn"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            {theme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
          </button>
          {isAuthenticated ? (
            <ProfileMenu />
          ) : (
            <Link to="/login">
              <Button size="sm">Log in</Button>
            </Link>
          )}
          <button
            type="button"
            className="ui-icon-btn ui-hamburger"
            onClick={() => setMenuOpen((value) => !value)}
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={menuOpen}
          >
            {menuOpen ? <IconX size={18} /> : <IconMenu size={18} />}
          </button>
        </div>
      </div>
    </header>
  );
};

export default AppHeader;
