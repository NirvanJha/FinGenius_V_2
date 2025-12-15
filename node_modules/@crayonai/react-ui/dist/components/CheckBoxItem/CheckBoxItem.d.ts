import { CSSProperties, ReactNode } from "react";
export interface CheckBoxItemProps {
    label?: ReactNode;
    description?: ReactNode;
    className?: string;
    style?: CSSProperties;
    checked?: boolean;
    defaultChecked?: boolean;
    disabled?: boolean;
    required?: boolean;
    name?: string;
    value?: string;
    onChange?: (checked: boolean) => void;
}
declare const CheckBoxItem: import("react").ForwardRefExoticComponent<CheckBoxItemProps & import("react").RefAttributes<HTMLButtonElement>>;
export { CheckBoxItem };
//# sourceMappingURL=CheckBoxItem.d.ts.map