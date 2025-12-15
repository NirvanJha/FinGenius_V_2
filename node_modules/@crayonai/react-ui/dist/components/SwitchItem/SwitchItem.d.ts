import { CSSProperties, ReactNode } from "react";
interface SwitchItemProps {
    label?: ReactNode;
    description?: ReactNode;
    className?: string;
    style?: CSSProperties;
    checked?: boolean;
    disabled?: boolean;
    defaultChecked?: boolean;
    required?: boolean;
    name?: string;
    value?: string;
    onChange?: (value: boolean) => void;
}
declare const SwitchItem: import("react").ForwardRefExoticComponent<SwitchItemProps & import("react").RefAttributes<HTMLButtonElement>>;
export { SwitchItem, type SwitchItemProps };
//# sourceMappingURL=SwitchItem.d.ts.map