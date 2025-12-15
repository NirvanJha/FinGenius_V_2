import React from "react";
import { CheckBoxItemProps } from "../CheckBoxItem";
type CheckBoxGroupVariant = "clear" | "card" | "sunk";
interface CheckBoxGroupProps {
    children: React.ReactElement<CheckBoxItemProps> | React.ReactElement<CheckBoxItemProps>[];
    className?: string;
    style?: React.CSSProperties;
    variant?: CheckBoxGroupVariant;
}
declare const CheckBoxGroup: React.ForwardRefExoticComponent<CheckBoxGroupProps & React.RefAttributes<HTMLDivElement>>;
export { CheckBoxGroup, type CheckBoxGroupProps };
//# sourceMappingURL=CheckBoxGroup.d.ts.map