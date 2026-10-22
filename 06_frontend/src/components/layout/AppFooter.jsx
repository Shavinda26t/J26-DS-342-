import React from 'react';
import { Link } from 'react-router-dom';
import { IconMail, IconMapPin, IconPhone, IconShieldCheck } from '../common/Icons';

const QUICK_LINKS = [
  { to: '/', label: 'Home' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/about', label: 'About' },
  { to: '/contact', label: 'Contact' },
];

const MODULE_LINKS = [
  { to: '/fleet', label: 'Fleet Intelligence (C1)' },
  { to: '/digital-twin', label: 'Digital Twin (C2)' },
  { to: '/causal-xai', label: 'Causal XAI (C3)' },
  { to: '/safe-rl', label: 'Safe Optimization (C4)' },
];

const AppFooter = () => (
  <footer className="ui-footer">
    <div className="ui-container">
      <div className="ui-footer__grid">
        <div>
          <Link to="/" className="ui-brand">
            <span className="ui-brand__mark">
              <IconShieldCheck size={18} />
            </span>
            <span className="ui-brand__text">HDD Health Framework</span>
          </Link>
          <p className="ui-footer__desc">
            An AI-based HDD failure prediction and self-healing framework combining behavioural drift
            detection, digital twins, causal explainability and safe reinforcement learning.
          </p>
        </div>

        <div>
          <div className="ui-footer__heading">Quick Links</div>
          {QUICK_LINKS.map((link) => (
            <Link key={link.to} to={link.to} className="ui-footer__link">
              {link.label}
            </Link>
          ))}
        </div>

        <div>
          <div className="ui-footer__heading">Contact</div>
          <div className="ui-footer__contact-item">
            <IconMail size={16} />
            research@example.edu
          </div>
          <div className="ui-footer__contact-item">
            <IconPhone size={16} />
            +00 000 000 0000
          </div>
          <div className="ui-footer__contact-item">
            <IconMapPin size={16} />
            University Research Lab
          </div>
        </div>
      </div>

      <div className="ui-footer__bottom">
        <span>© {new Date().getFullYear()} HDD Failure Prediction &amp; Self-Healing Framework. All rights reserved.</span>
        <span>Final-year research project</span>
      </div>
    </div>
  </footer>
);

export default AppFooter;
