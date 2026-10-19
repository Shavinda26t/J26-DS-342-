import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { IconChevronDown, IconLayoutGrid, IconLogOut } from './Icons';

const getInitials = (name = '') =>
  name
    .split(' ')
    .map((part) => part[0])
    .filter(Boolean)
    .slice(0, 2)
    .join('')
    .toUpperCase();

const ProfileMenu = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  if (!user) return null;

  const goTo = (path) => {
    setOpen(false);
    navigate(path);
  };

  return (
    <div className="ui-profile" ref={containerRef}>
      <button
        type="button"
        className="ui-profile__btn"
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className="ui-avatar">{getInitials(user.name)}</span>
        <span className="ui-profile__name">{user.name}</span>
        <IconChevronDown size={16} />
      </button>
      {open && (
        <div className="ui-profile__menu" role="menu">
          <div className="ui-menu__header">
            <div className="ui-menu__name">{user.name}</div>
            <div className="ui-menu__email">{user.email}</div>
          </div>
          <button type="button" className="ui-menu__item" role="menuitem" onClick={() => goTo('/dashboard')}>
            <IconLayoutGrid size={16} />
            Dashboard
          </button>
          <button
            type="button"
            className="ui-menu__item ui-menu__item--danger"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              logout();
              navigate('/');
            }}
          >
            <IconLogOut size={16} />
            Log out
          </button>
        </div>
      )}
    </div>
  );
};

export default ProfileMenu;
