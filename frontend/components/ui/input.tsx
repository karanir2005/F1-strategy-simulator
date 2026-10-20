import * as React from "react"
import { Input as InputPrimitive } from "@base-ui/react/input"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <InputPrimitive
      type={type}
      data-slot="input"
      className={cn(
        "h-9 w-full min-w-0 rounded-lg border border-line-strong bg-sunken px-2.5 py-1 font-mono text-sm tnum text-fg transition-colors outline-none placeholder:text-faint focus-visible:border-signal focus-visible:ring-2 focus-visible:ring-signal/40 disabled:cursor-not-allowed disabled:opacity-50 aria-[invalid=true]:border-bad",
        className
      )}
      {...props}
    />
  )
}

export { Input }
