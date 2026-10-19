import React from 'react';

const StatCard = ({ label, value, delta = null, deltaDirection = 'up', tone = 'blue', icon = null }) => (
  <div className="ui-stat">
    {icon && <div className={`ui-stat__icon ${tone !== 'blue' ? `ui-stat__icon--${tone}` : ''}`.trim()}>{icon}</div>}
    <div>
      <div className="ui-stat__label">{label}</div>
      <div className="ui-stat__value">{value}</div>
      {delta && <div className={`ui-stat__delta ui-stat__delta--${deltaDirection}`}>{delta}</div>}
    </div>
  </div>
);

export default StatCard;
