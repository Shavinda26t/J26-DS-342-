import React from 'react';
import { Link } from 'react-router-dom';
import Button from '../components/common/Button';
import { IconArrowRight, IconHome } from '../components/common/Icons';

const NotFound = () => (
  <div className="ui-404">
    <div className="ui-404__code ui-grad-text">404</div>
    <h1 className="ui-404__title">Page not found</h1>
    <p className="ui-404__sub">
      The page you are looking for does not exist or may have been moved. Head back to safety.
    </p>
    <div className="ui-404__actions">
      <Link to="/">
        <Button leftIcon={<IconHome size={16} />}>
          Back to home
        </Button>
      </Link>
      <Link to="/dashboard">
        <Button variant="secondary" rightIcon={<IconArrowRight size={16} />}>
          Open dashboard
        </Button>
      </Link>
    </div>
  </div>
);

export default NotFound;
