import { mergeProps } from "@base-ui/react/merge-props"
import { useRender } from "@base-ui/react/use-render"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex h-5 w-fit shrink-0 items-center justify-center gap-1 rounded-sm border border-transparent px-1.5 font-mono text-[11px] font-medium tracking-wide whitespace-nowrap uppercase",
  {
    variants: {
      variant: {
        default: "bg-signal text-signal-fg",
        secondary: "bg-hover text-fg",
        outline: "border-line-strong text-muted-fg",
        warn: "border-warn/50 bg-warn/10 text-warn",
        good: "border-good/40 bg-good/10 text-good",
        best: "border-best/40 bg-best/10 text-best",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  render,
  ...props
}: useRender.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return useRender({
    defaultTagName: "span",
    props: mergeProps<"span">(
      {
        className: cn(badgeVariants({ variant }), className),
      },
      props
    ),
    render,
    state: {
      slot: "badge",
      variant,
    },
  })
}

export { Badge, badgeVariants }
