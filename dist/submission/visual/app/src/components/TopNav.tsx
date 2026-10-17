export type Page = 'Monitor' | 'Events' | 'Model';

type Props = { page: Page; onPage: (page: Page) => void; demo: boolean };

export default function TopNav({ page, onPage, demo }: Props) {
  return (
    <header className="top-nav">
      <div className="brand"><strong>EatingSense</strong><small>Wrist IMU Eating Event Detection</small></div>
      <nav>{(['Monitor', 'Events', 'Model'] as const).map(x => <button key={x} className={page === x ? 'active' : ''} onClick={() => onPage(x)}>{x}</button>)}</nav>
      <div className="top-meta"><span className="offline-dot" />{demo ? 'DEMO DATA' : 'Offline analysis'}<span className="nav-time">{new Date().toLocaleDateString('en-CA')} &nbsp; {new Date().toLocaleTimeString('en-GB', { hour12: false })}</span></div>
    </header>
  );
}
