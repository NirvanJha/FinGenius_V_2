import * as SliderPrimitive from "@radix-ui/react-slider";
import { ReactNode } from "react";
export interface SliderProps extends Omit<React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>, "value" | "defaultValue"> {
    variant: "continuous" | "discrete";
    min: number;
    max: number;
    step?: number;
    disabled?: boolean;
    value?: number[];
    defaultValue?: number[];
    className?: string;
    style?: React.CSSProperties;
    leftContent?: ReactNode;
    rightContent?: ReactNode;
}
export declare const Slider: import("react").ForwardRefExoticComponent<SliderProps & import("react").RefAttributes<HTMLSpanElement>>;
//# sourceMappingURL=Slider.d.ts.map