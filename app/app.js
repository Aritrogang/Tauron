// Main App Entry
const { useState, useEffect } = React;

const App = () => {
    const [currentTab, setCurrentTab] = useState('home');
    const [showLandingPage, setShowLandingPage] = useState(true);

    useEffect(() => {
        if (window.lucide) {
            window.lucide.createIcons();
        }
    });

    const renderContent = () => {
        switch (currentTab) {
            case 'feed': return <MorningAlertFeed />;
            case 'map': return <HerdMap />;
            case 'log': return <DataEntryLog />;
            case 'impact': return <SustainabilityImpact />;
            case 'about': return <TierAbout />;
            case 'home': return <Homepage onNavigate={setCurrentTab} />;
            default: return <Homepage onNavigate={setCurrentTab} />;
        }
    };

    if (showLandingPage) {
        return (
            <div style={{
                position: 'fixed',
                top: 0, left: 0, right: 0, bottom: 0,
                zIndex: 9999,
                background: 'rgba(0, 0, 0, 0.85)',
                backdropFilter: 'blur(24px)',
                WebkitBackdropFilter: 'blur(24px)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                animation: 'slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards'
            }}>
                <div className="logo" style={{ fontSize: '96px', color: '#fff', marginBottom: '48px' }}>
                    Tauron<span className="accent" style={{ color: 'var(--sage)' }}>.</span>
                </div>

                <button
                    onClick={() => setShowLandingPage(false)}
                    className="hover-lift"
                    style={{
                        padding: '16px 56px',
                        fontSize: '20px',
                        fontFamily: 'Cormorant Garamond, serif',
                        fontWeight: '700',
                        letterSpacing: '0.05em',
                        color: '#fff',
                        background: 'var(--sage)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '40px',
                        cursor: 'pointer',
                        boxShadow: '0 12px 32px rgba(106, 158, 72, 0.4)',
                        transition: 'all 0.3s ease',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px'
                    }}
                >
                    GET STARTED
                    <i data-lucide="arrow-right" style={{ width: '20px', height: '20px' }}></i>
                </button>

                <p style={{
                    marginTop: '48px',
                    fontFamily: 'Cormorant Garamond, serif',
                    fontSize: '22px',
                    color: 'rgba(255, 255, 255, 0.6)',
                    maxWidth: '500px',
                    textAlign: 'center',
                    lineHeight: 1.5,
                    opacity: 0.8
                }}>
                    GNN-powered disease prediction <br />
                    48 hours before symptoms appear.
                </p>
            </div>
        );
    }

    // Homepage renders full-width without the sidebar Layout
    if (currentTab === 'home') {
        return <Homepage onNavigate={setCurrentTab} />;
    }

    return (
        <Layout currentTab={currentTab} onTabChange={setCurrentTab}>
            {renderContent()}
        </Layout>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
