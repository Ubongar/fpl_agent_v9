import Link from "next/link";

export default function NotFound() {
  return (
    <div className="py-16 text-center">
      <h1 className="text-xl font-semibold">Page not found</h1>
      <Link href="/" className="mt-2 inline-block text-sm text-[var(--accent)]">
        Back to projections
      </Link>
    </div>
  );
}
