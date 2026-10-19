import React, { useState } from 'react';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import Card from '../components/common/Card';
import Reveal from '../components/common/Reveal';
import { IconCheck, IconMail, IconMapPin, IconPhone, IconSend } from '../components/common/Icons';

const INITIAL_FORM = { name: '', email: '', subject: '', message: '' };

const validate = (form) => {
  const errors = {};
  if (!form.name.trim()) errors.name = 'Name is required.';
  if (!form.email.trim()) {
    errors.email = 'Email is required.';
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) {
    errors.email = 'Enter a valid email address.';
  }
  if (!form.subject.trim()) errors.subject = 'Subject is required.';
  if (!form.message.trim()) {
    errors.message = 'Message is required.';
  } else if (form.message.trim().length < 10) {
    errors.message = 'Message must be at least 10 characters.';
  }
  return errors;
};

const CONTACT_ITEMS = [
  { icon: IconMail, label: 'Email', value: 'research@example.edu' },
  { icon: IconPhone, label: 'Phone', value: '+00 000 000 0000' },
  { icon: IconMapPin, label: 'Location', value: 'University Research Lab' },
];

const Contact = () => {
  const [form, setForm] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [sent, setSent] = useState(false);

  const updateField = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
    if (errors[name]) {
      setErrors((current) => ({ ...current, [name]: undefined }));
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const nextErrors = validate(form);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) return;
    setSent(true);
    setForm(INITIAL_FORM);
  };

  return (
    <>
      <section className="ui-page-hero">
        <div className="ui-container">
          <span className="ui-eyebrow">Contact</span>
          <h1 className="ui-page-hero__title">Get in touch</h1>
          <p className="ui-page-hero__sub">
            Questions about the framework, collaboration opportunities or supervision — drop a message and the
            research team will get back to you.
          </p>
        </div>
      </section>

      <section className="ui-section" style={{ paddingTop: 32 }}>
        <div className="ui-container">
          <div className="ui-contact-grid">
            <Reveal style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {CONTACT_ITEMS.map((item) => {
                const Icon = item.icon;
                return (
                  <div key={item.label} className="ui-contact-item">
                    <div className="ui-contact-item__icon">
                      <Icon size={18} />
                    </div>
                    <div>
                      <div className="ui-contact-item__label">{item.label}</div>
                      <div className="ui-contact-item__value">{item.value}</div>
                    </div>
                  </div>
                );
              })}
              <div className="ui-alert ui-alert--info">
                <IconMail size={16} />
                Contact details are placeholders for this research prototype.
              </div>
            </Reveal>

            <Reveal>
              <Card title="Send a message">
                {sent ? (
                  <div className="ui-alert ui-alert--success" role="status">
                    <IconCheck size={16} />
                    Message captured (demo mode — nothing was sent to a server). Thank you!
                  </div>
                ) : (
                  <form className="ui-form" onSubmit={handleSubmit} noValidate>
                    <div className="ui-form__row">
                      <Input
                        label="Name"
                        name="name"
                        placeholder="Your name"
                        value={form.name}
                        onChange={updateField}
                        error={errors.name}
                      />
                      <Input
                        label="Email"
                        type="email"
                        name="email"
                        placeholder="you@example.com"
                        value={form.email}
                        onChange={updateField}
                        error={errors.email}
                      />
                    </div>
                    <Input
                      label="Subject"
                      name="subject"
                      placeholder="What is this about?"
                      value={form.subject}
                      onChange={updateField}
                      error={errors.subject}
                    />
                    <div className="ui-field">
                      <label className="ui-label" htmlFor="contact-message">
                        Message
                      </label>
                      <textarea
                        id="contact-message"
                        name="message"
                        rows={5}
                        className="ui-input"
                        placeholder="Write your message…"
                        value={form.message}
                        onChange={updateField}
                        aria-invalid={errors.message ? 'true' : undefined}
                        style={{ resize: 'vertical', minHeight: 120 }}
                      />
                      {errors.message && (
                        <span className="ui-error-text" role="alert">
                          {errors.message}
                        </span>
                      )}
                    </div>
                    <Button type="submit" leftIcon={<IconSend size={16} />}>
                      Send message
                    </Button>
                  </form>
                )}
              </Card>
            </Reveal>
          </div>
        </div>
      </section>
    </>
  );
};

export default Contact;
