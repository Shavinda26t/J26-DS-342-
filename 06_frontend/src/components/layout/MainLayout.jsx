import React from 'react';
import { Outlet } from 'react-router-dom';
import AppHeader from './AppHeader';
import AppFooter from './AppFooter';

const MainLayout = () => (
  <div className="ui-app">
    <AppHeader />
    <main className="ui-main">
      <Outlet />
    </main>
    <AppFooter />
  </div>
);

export default MainLayout;
