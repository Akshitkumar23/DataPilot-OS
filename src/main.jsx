import React, { Component, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("DataPilot OS Crash Caught by Error Boundary:", error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          height: '100vh',
          width: '100vw',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#090d16',
          color: '#f8fafc',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          padding: '24px',
          boxSizing: 'border-box'
        }}>
          <div style={{
            maxWidth: '600px',
            width: '100%',
            background: '#0f172a',
            border: '1px solid #1e293b',
            borderRadius: '12px',
            padding: '32px',
            boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
            <h1 style={{ fontSize: '20px', fontWeight: 800, color: '#ef4444', marginBottom: '8px' }}>
              DataPilot OS System Exception
            </h1>
            <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '20px', lineHeight: 1.6 }}>
              An unhandled rendering error occurred. You can attempt to reload the application shell or inspect the runtime log.
            </p>

            {this.state.error && (
              <div style={{
                background: '#020617',
                border: '1px solid #334155',
                borderRadius: '6px',
                padding: '12px',
                marginBottom: '20px',
                fontSize: '11px',
                fontFamily: 'monospace',
                color: '#fca5a5',
                textAlign: 'left',
                overflowX: 'auto',
                maxHeight: '150px'
              }}>
                {this.state.error.toString()}
              </div>
            )}

            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
              <button
                onClick={this.handleReload}
                style={{
                  background: '#0284c7',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  fontSize: '12px',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'background 0.2s'
                }}
              >
                🔄 Reload Application
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
