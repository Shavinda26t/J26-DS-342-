import React from 'react';
import { Link, Outlet } from 'react-router-dom';
import { IconShieldCheck } from '../common/Icons';

const AuthLayout = () => (
  <div className="ui-app ui-auth-layout">
    <header className="ui-auth__topbar">
      <Link to="/" className="ui-brand">
        <span className="ui-brand__mark">
          <IconShieldCheck size={18} />
        </span>
        <span className="ui-brand__text">HDD Health Framework</span>
      </Link>
      <Link to="/" className="ui-auth__back">
        ← Back to home
      </Link>
    </header>
    <main className="ui-auth__main">
      <Outlet />
    </main>
    <footer className="ui-auth__site-footer">
      © {new Date().getFullYear()} HDD Failure Prediction &amp; Self-Healing Framework
    </footer>
  </div>
);

export default AuthLayout;
