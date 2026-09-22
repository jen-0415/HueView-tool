export default function Card({ children, className = "" }) {
  return (
    <div className={`bg-white border border-line-soft rounded-2xl ${className}`}>
      {children}
    </div>
  );
}
  