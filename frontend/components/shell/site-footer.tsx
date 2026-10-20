export function SiteFooter() {
  return (
    <footer className="mt-20 border-t border-line">
      <div className="mx-auto max-w-[1200px] space-y-1 px-4 py-6 text-[13px] leading-5 text-muted-fg sm:px-6">
        <p>
          Data: Jolpica-F1 results and OpenF1 timing, with earlier races from FastF1. Predictions are probabilities from
          a statistical model, not picks or betting advice.
        </p>
        <p>
          Strategy times come from a one-car Monte Carlo simulation (no safety cars, traffic or weather). They are good
          for comparing plans, not for quoting race times.
        </p>
      </div>
    </footer>
  );
}
