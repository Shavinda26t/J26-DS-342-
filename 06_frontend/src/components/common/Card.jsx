import React from 'react';

const Card = ({ title, actions, footer, hover = false, className = '', bodyClassName = '', children }) => (
  <div className={`ui-card ${hover ? 'ui-card--hover' : ''} ${className}`.trim()}>
    {(title || actions) && (
      <div className="ui-card__header">
        {title && <h3 className="ui-card__title">{title}</h3>}
        {actions && <div className="ui-card__actions">{actions}</div>}
      </div>
    )}
    <div className={`ui-card__body ${bodyClassName}`.trim()}>{children}</div>
    {footer && <div className="ui-card__footer">{footer}</div>}
  </div>
);

export default Card;
