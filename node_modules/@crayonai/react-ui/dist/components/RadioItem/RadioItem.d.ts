import { CSSProperties, ReactNode } from "react";
export interface RadioItemProps {
    label?: ReactNode;
    description?: ReactNode;
    className?: string;
    style?: CSSProperties;
    disabled?: boolean;
    required?: boolean;
    value: string;
}
declare const RadioItem: import("react").ForwardRefExoticComponent<RadioItemProps & import("react").RefAttributes<HTMLButtonElement>>;
export { RadioItem };
//# sourceMappingURL=RadioItem.d.ts.map