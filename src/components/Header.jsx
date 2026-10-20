import React from 'react';
import TitleBar from './TitleBar';
import Ribbon from './Ribbon';

export default function Header({
  datasetName,
  rowCount,
  colCount,
  activeModule,
  onModuleChange,
  datasetLoaded,
  onOpenFile,
  onOpenDbModal,
  onSaveFile,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  isSaving,
  onOpenCodeInspector,
  isRibbonCollapsed,
  onToggleCollapse,
  theme,
  onToggleTheme,
  onNavigateHome
}) {
  return (
    <header className="app-header" style={{ display: 'flex', flexDirection: 'column', flexShrink: 0, width: '100%' }}>
      <TitleBar
        datasetName={datasetName}
        rowCount={rowCount}
        colCount={colCount}
        onOpenFile={onOpenFile}
        onSaveFile={onSaveFile}
        isSaving={isSaving}
        theme={theme}
        onToggleTheme={onToggleTheme}
        onNavigateHome={onNavigateHome}
      />
      <Ribbon
        activeModule={activeModule}
        onModuleChange={onModuleChange}
        datasetLoaded={datasetLoaded}
        onOpenFile={onOpenFile}
        onOpenDbModal={onOpenDbModal}
        onSaveFile={onSaveFile}
        onUndo={onUndo}
        onRedo={onRedo}
        canUndo={canUndo}
        canRedo={canRedo}
        isSaving={isSaving}
        onOpenCodeInspector={onOpenCodeInspector}
        isCollapsed={isRibbonCollapsed}
        onToggleCollapse={onToggleCollapse}
      />
    </header>
  );
}

export { TitleBar, Ribbon };
